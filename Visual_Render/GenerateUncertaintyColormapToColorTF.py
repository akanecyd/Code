import os
import matplotlib.cm as cm
import numpy as np

def write_colormap_to_file(pth, vals, colors):
    with open(pth,'w') as fw:
        for idx, _val in enumerate(vals):
            fw.write('%.5f, %.3f,%.3f,%.3f\n'\
                    %(_val, colors[idx,0],colors[idx,1],colors[idx,2]))

def write_opacity_to_file(pth, vals,th, upper,lower):
    with open(pth,'w') as fw:
        for idx, _val in enumerate(vals):
            if _val > th:
                fw.write('%.5f, %.5f\n'\
                        %(_val, upper))
            else:
                fw.write('%.5f, %.5f\n'\
                        %(_val, lower))


# Color
_NAME = 'jet'
_COLORMAP_ROOT = './TFs'
_COL_RES = 64
_OP_RES = 4

# Opacity
_TH = 0.0005
_UPPER = 0.03
_LOWER = 0.00005

#Set range of values
lut = np.linspace(0, 1, _COL_RES)
col_vals = np.linspace(0, 0.01, _COL_RES)
op_vals = np.linspace(0, 0.01, _OP_RES)

#Generate colormap
_cmap = cm.get_cmap(_NAME)
#Generate colors 
_colors = _cmap(lut)

#Write to file
write_colormap_to_file(os.path.join(_COLORMAP_ROOT, 'Uncertainty_color_tf.txt'),
                       col_vals,
                       _colors)

write_opacity_to_file(os.path.join(_COLORMAP_ROOT, 'Uncertainty_opacity_tf.txt'),
                      op_vals,_TH, _UPPER,_LOWER)


