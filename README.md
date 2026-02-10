Tanuj Tekkale (tt27868), Akshay Gaitonde (ag84839)

__Assignment Part 1__
- We determine whether a ray is entering or exiting an object by checking the sign of the dot product between the ray direction and the surface normal. If the dot product is zero, it is treated as entering.
- Shadow rays support partial transparency. What this means is that when a shadow ray intersects a translucent object, the light contribution is attenuated by the material's transmissive coefficient instead of being fully blocked.
- When spawning secondary rays, we offset the ray origin by a small epsilon along the ray's direction rather than the surface normal to avoid issues with self-intersection.
- We do not implement any form of backface culling and instead treat all geometry as two-sided. This means that rays are allowed to intersect both the front and back faces for primary, shadow, reflection, and refraction rays.
- Triangle meshes support per-vertex normals, colors, and texture coordinates, which are interpolated across triangle surfaces during intersection and shading. Degenerate triangles are ignored.
- Directional lights do not apply distance attenuation, since their light rays are assumed to originate from an infinitely distant source.
