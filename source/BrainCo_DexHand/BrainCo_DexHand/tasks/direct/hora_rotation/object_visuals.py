"""Color blocks for rotation visibility, separate from the original analytic collider."""
import math

from pxr import Sdf, UsdGeom, UsdShade

import isaaclab.sim as sim_utils


_COLORS = [
    (0.75, 0.025, 0.025), (0.02, 0.45, 0.65), (0.90, 0.55, 0.02),
    (0.05, 0.55, 0.12), (0.36, 0.04, 0.70), (0.85, 0.12, 0.02),
]


@sim_utils.clone
def spawn_rotation_object(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the usual sphere/cylinder physics and add a non-colliding colored surface."""
    sphere = isinstance(cfg, sim_utils.SphereCfg)
    if not sphere and (not isinstance(cfg, sim_utils.CylinderCfg) or cfg.axis != "Z"):
        raise ValueError("Rotation visuals support spheres and Z-axis cylinders")
    spawn = sim_utils.spawn_sphere if sphere else sim_utils.spawn_cylinder
    root = spawn(prim_path, cfg, translation, orientation, **kwargs)
    stage = root.GetStage()
    # Visibility does not disable collision; keep the original shape and its material untouched.
    UsdGeom.Imageable(stage.GetPrimAtPath(prim_path + "/geometry/mesh")).MakeInvisible()
    mesh = UsdGeom.Mesh.Define(stage, prim_path + "/geometry/color_blocks")
    points, normals, counts, colors = [], [], [], []

    def face(vertices, vertex_normals, color):
        points.extend(vertices)
        normals.extend(vertex_normals)
        counts.append(len(vertices))
        colors.append(_COLORS[color % len(_COLORS)])

    segments = 48  # Eight faces per color sector; normals keep the surface smooth.
    bands = 16 if sphere else 2
    for band in range(bands):
        for sector in range(segments):
            angles = [2 * math.pi * i / segments for i in (sector, sector + 1)]
            if sphere:
                latitudes = [-math.pi / 2 + math.pi * j / bands for j in (band, band + 1)]
                ns = [(math.cos(lat) * math.cos(angle), math.cos(lat) * math.sin(angle), math.sin(lat))
                      for lat, angle in ((latitudes[0], angles[0]), (latitudes[0], angles[1]),
                                         (latitudes[1], angles[1]), (latitudes[1], angles[0]))]
                # One pole vertex per triangle, rather than degenerate quads at the poles.
                if band == 0:
                    ns = [ns[0], ns[2], ns[3]]
                elif band == bands - 1:
                    ns = ns[:3]
                face([tuple(cfg.radius * v for v in n) for n in ns], ns, sector // 8 + 2 * (band * 3 // bands))
            else:
                zs = [cfg.height * (j / bands - 0.5) for j in (band, band + 1)]
                ns = [(math.cos(a), math.sin(a), 0.0) for a in (angles[0], angles[1], angles[1], angles[0])]
                vertices = [(cfg.radius * n[0], cfg.radius * n[1], z) for n, z in zip(ns, [zs[0], zs[0], zs[1], zs[1]])]
                face(vertices, ns, sector // 8 + 3 * band)
                if band == 0:
                    face([(0.0, 0.0, zs[0]), vertices[1], vertices[0]], [(0.0, 0.0, -1.0)] * 3, sector // 8)
                else:
                    face([(0.0, 0.0, zs[1]), vertices[3], vertices[2]], [(0.0, 0.0, 1.0)] * 3, sector // 8 + 3)

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(colors)

    material = UsdShade.Material.Define(stage, prim_path + "/geometry/color_material")
    reader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Color")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return root
