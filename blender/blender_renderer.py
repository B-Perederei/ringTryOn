import bpy
from pathlib import Path
from mathutils import Matrix
import numpy as np


class BlenderRenderSetup:
    def __init__(self, p_root: Path, dn_ext="OPEN_EXR"):
        assert dn_ext in ["OPEN_EXR", "PNG"]

        self.camera_name: str = None
        self.camera = None
        self.tree = None

        self.p_root = p_root
        self.p_out_render = None
        self.dn_ext = dn_ext.upper()

    def _set_camera(
            self,
            camera_name: str,
    ) -> None:
        self.camera = bpy.data.objects.get(camera_name)
        assert self.camera is not None
        self.camera_name = camera_name

    def _set_render_path(
            self
    ) -> None:
        assert self.camera_name
        self.p_out_render = Path(self.p_root) / self.camera_name
        self.p_out_render.mkdir(exist_ok=True, parents=True)

    def _set_context(
            self,
            px: int = 1024,
            py: int = 1024,
    ) -> None:
        assert self.camera is not None
        bpy.context.scene.camera = self.camera
        bpy.context.scene.render.resolution_x = px
        bpy.context.scene.render.resolution_y = py
        bpy.context.scene.use_nodes = True

        bpy.context.view_layer.use_pass_z = True
        bpy.context.view_layer.use_pass_normal = True
        bpy.context.view_layer.use_pass_object_index = True
        bpy.context.scene.render.film_transparent = True

    def _get_output_node(
            self,
            tree,
            label,
            file_format: str,
            use_alpha: str = False
    ):
        base_path = self.p_out_render / f"_{label}"
        base_path.mkdir(exist_ok=True, parents=True)

        node = tree.nodes.new(type="CompositorNodeOutputFile")
        node.base_path = str(base_path)
        node.format.file_format = file_format

        if use_alpha:
            node.format.color_mode = 'RGBA'

        return node

    def _set_nodes_tree(
            self
    ) -> None:
        self.tree = bpy.context.scene.node_tree
        self.tree.nodes.clear()

        self.render_layers = self.tree.nodes.new(type="CompositorNodeRLayers")
        self.composite = self.tree.nodes.new(type="CompositorNodeComposite")

        self.file_output_color = self._get_output_node(self.tree, "color", "PNG", use_alpha=True)
        self.file_output_depth = self._get_output_node(self.tree, "depth", self.dn_ext)
        self.file_output_normal = self._get_output_node(self.tree, "normal", self.dn_ext)
    def _link_nodes(self):
        assert self.tree is not None

        print(self.render_layers.outputs)
        self.tree.links.new(self.render_layers.outputs["Image"], self.file_output_color.inputs[0])
        self.tree.links.new(self.render_layers.outputs["Depth"], self.file_output_depth.inputs[0])
        self.tree.links.new(self.render_layers.outputs["Normal"], self.file_output_normal.inputs[0])
        # self.tree.links.new(self.render_layers.outputs["IndexOB"], self.file_output_mask.inputs[0])
        self.tree.links.new(self.render_layers.outputs["Image"], self.composite.inputs[0])

    def _transform_matrix_to_euler(self, transform_matrix):
        rotation_matrix = transform_matrix[:3, :3]
        location = transform_matrix[:3, 3]
        euler_angles = Matrix(rotation_matrix).to_euler()
        return euler_angles, location

    def _setup(
            self,
            camera_name: str,
            px: int = 1024,
            py: int = 1024,
    ) -> None:
        self._set_camera(camera_name)
        self._set_render_path()
        self._set_context(px=px, py=py)
        self._set_nodes_tree()
        self._link_nodes()

        transform_matrix = np.array([[-1.83043312e-01, -9.01450335e-01, -3.92278524e-01,  2.99722454e+01],
                                     [-9.61303787e-01,  8.05525760e-02,  2.63450778e-01,  3.63439362e+01],
                                     [-2.05888746e-01,  4.25321733e-01, -8.81312230e-01,  3.40888294e+02],
                                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
                                     ])

        m2 = np.array([[1, 0, 0,  -0.06836785606168191],
                       [0, 1, 0, -0.2043487541167911],
                       [0, 0, 1,  0],
                       [0, 0, 0,   1]])
        euler_angles, location = self._transform_matrix_to_euler(transform_matrix)
        euler_camera, camera_location = self._transform_matrix_to_euler(m2)

        # bpy.data.objects["ring"].location = (location[1], location[0], location[2])
        # bpy.data.objects["ring"].rotation_euler = euler_angles
        # bpy.data.objects["bottom"].location = camera_location
        # bpy.data.objects["bottom"].rotation_euler = euler_camera

    def _render(
            self
    ) -> None:
        bpy.ops.render.render(write_still=True)
        print(f"Rendered color, depth, and normal maps saved to {self.p_out_render}")

    def render(
            self,
            camera_name: str,
            px: int = 1920,
            py: int = 1440,
    ) -> None:
        self._setup(
            camera_name=camera_name,
            px=px, py=py
        )

        assert self.camera is not None
        assert self.tree is not None

        self._render()


if __name__ == '__main__':
    p_root = Path(r"./results")

    renderer = BlenderRenderSetup(p_root)
    for view in ['bottom', 'front']:
        renderer.render(view)

