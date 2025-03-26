import os
import cv2
import json
import torch
import trimesh
import kaolin as kal


class Mesh:
    def __init__(self, mesh_path, device, target_scale=1.0, mesh_dy=0.0,
                 remove_mesh_part_names=None, remove_unsupported_buffers=None, intermediate_dir=None):
        # from https://github.com/threedle/text2mesh
        self.material_cvt, self.material_num, org_mesh_path, is_convert = None, 1, mesh_path, False
        self.vt, self.ft = None, None
        if not mesh_path.endswith(".obj") and not mesh_path.endswith(".off"):
            if mesh_path.endswith(".gltf"):
                mesh_path = self.preprocess_gltf(mesh_path, remove_mesh_part_names, remove_unsupported_buffers)
            mesh_temp = trimesh.load(mesh_path, force='mesh', process=True, maintain_order=True)
            mesh_path = os.path.splitext(mesh_path)[0] + "_cvt.obj"
            mesh_temp.export(mesh_path)
            merge_texture_path = os.path.join(os.path.dirname(mesh_path), "material_0.png")
            if os.path.exists(merge_texture_path):
                self.material_cvt = cv2.imread(merge_texture_path)
                self.material_num = self.material_cvt.shape[1] // self.material_cvt.shape[0]
            print("Converting current mesh model to obj file with {} material~".format(self.material_num))
            is_convert = True

        if ".obj" in mesh_path:
            # Load with trimesh first to get UV coordinates
            trimesh_mesh = trimesh.load(mesh_path, process=False, maintain_order=True)

            # Get UV coordinates from trimesh
            if hasattr(trimesh_mesh, 'visual') and hasattr(trimesh_mesh.visual, 'texture'):
                # Get UV coordinates and faces
                self.vt = torch.from_numpy(trimesh_mesh.visual.uv).float().to(device)
                if hasattr(trimesh_mesh.visual, 'face_texture'):
                    self.ft = torch.from_numpy(trimesh_mesh.visual.face_texture).long().to(device)
                else:
                    self.ft = torch.from_numpy(trimesh_mesh.faces).long().to(device)

                # Print detailed UV information
                print("[Paint3D] UV Coordinates Info:")
                print(f"[Paint3D] UV Min X: {self.vt[:,0].min():.4f}, Max X: {self.vt[:,0].max():.4f}")
                print(f"[Paint3D] UV Min Y: {self.vt[:,1].min():.4f}, Max Y: {self.vt[:,1].max():.4f}")
                print(f"[Paint3D] UV Face indices range: {self.ft.min()}-{self.ft.max()}")

            # Load geometry with kaolin
            try:
                mesh = kal.io.obj.import_mesh(mesh_path, with_normals=True, with_materials=True)
            except:
                mesh = kal.io.obj.import_mesh(mesh_path, with_normals=True, with_materials=False)

            self.vertices = mesh.vertices.to(device)
            self.faces = mesh.faces.to(device)

        elif ".off" in mesh_path:
            mesh = kal.io.off.import_mesh(mesh_path)
            self.vertices = mesh.vertices.to(device)
            self.faces = mesh.faces.to(device)
            self.vt = None
            self.ft = None
        else:
            raise ValueError(f"{mesh_path} extension not implemented in mesh reader.")

        self.mesh_path = mesh_path
        self.normalize_mesh(target_scale=target_scale, mesh_dy=mesh_dy)

        if is_convert and intermediate_dir is not None:
            if not os.path.exists(intermediate_dir):
                os.makedirs(intermediate_dir)
            if os.path.exists(os.path.splitext(org_mesh_path)[0] + "_removed.gltf"):
                os.system("mv {} {}".format(os.path.splitext(org_mesh_path)[0] + "_removed.gltf", intermediate_dir))
            if mesh_path.endswith("_cvt.obj"):
                os.system("mv {} {}".format(mesh_path, intermediate_dir))
            os.system("mv {} {}".format(os.path.join(os.path.dirname(mesh_path), "material.mtl"), intermediate_dir))
            if os.path.exists(merge_texture_path):
                os.system("mv {} {}".format(os.path.join(os.path.dirname(mesh_path), "material_0.png"), intermediate_dir))

    def has_valid_uv_mapping(self):
        """Check if the mesh has valid UV coordinates"""
        if self.vt is None or self.ft is None:
            print("[Paint3D] No UV coordinates found")
            return False
        if self.vt.shape[0] == 0:  # No UV vertices
            print("[Paint3D] Empty UV vertex array")
            return False
        if self.ft.min() <= -1:  # Invalid UV face indices
            print("[Paint3D] Invalid UV face indices found")
            return False

        print(f"[Paint3D] Valid UV mapping found: UV vertices={self.vt.shape}, UV faces={self.ft.shape}")
        return True

    def preprocess_gltf(self, mesh_path, remove_mesh_part_names, remove_unsupported_buffers):
        with open(mesh_path, "r") as fr:
            gltf_json = json.load(fr)
            if remove_mesh_part_names is not None:
                temp_primitives = []
                for primitive in gltf_json["meshes"][0]["primitives"]:
                    if_append, material_id = True, primitive["material"]
                    material_name = gltf_json["materials"][material_id]["name"]
                    for remove_mesh_part_name in remove_mesh_part_names:
                        if material_name.find(remove_mesh_part_name) >= 0:
                            if_append = False
                            break
                    if if_append:
                        temp_primitives.append(primitive)
                gltf_json["meshes"][0]["primitives"] = temp_primitives
                print("Deleting mesh with materials named '{}' from gltf model ~".format(remove_mesh_part_names))

            if remove_unsupported_buffers is not None:
                temp_buffers = []
                for buffer in gltf_json["buffers"]:
                    if_append = True
                    for unsupported_buffer in remove_unsupported_buffers:
                        if buffer["uri"].find(unsupported_buffer) >= 0:
                            if_append = False
                            break
                    if if_append:
                        temp_buffers.append(buffer)
                gltf_json["buffers"] = temp_buffers
                print("Deleting unspported buffers within uri {} from gltf model ~".format(remove_unsupported_buffers))
            updated_mesh_path = os.path.splitext(mesh_path)[0] + "_removed.gltf"
            with open(updated_mesh_path, "w") as fw:
                json.dump(gltf_json, fw, indent=4)
        return updated_mesh_path

    def normalize_mesh(self, target_scale=1.0, mesh_dy=0.0):
        verts = self.vertices
        self.original_center = verts.mean(dim=0)  # Store original center
        verts = verts - self.original_center
        self.original_scale = torch.max(torch.norm(verts, p=2, dim=1))  # Store original scale
        verts = verts / self.original_scale
        verts *= target_scale
        verts[:, 1] += mesh_dy
        self.vertices = verts

        # remove any rescaling
        # and translation of the mesh
        # pass
