import numpy as np

# Use like
#	write_ply(out_fn, out_points, out_colours)
#		out_fn = output file name (e.g. "out.ply")
#		out_points = 3D points
#		out_colours = colours of points
# Resulting .ply file can be viewed with MeshLab ( http://meshlab.sourceforge.net/ )
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
    print('%s saved' % fn)
