
from numpy import max, min, zeros
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def MatplotlibImshow3D(img_array, figsize, image_id=None):
    
    def on_press(event):
        if (event.xdata and event.ydata):
            if (event.button == "up"):
                if not (s_slide.val == idx_max):
                    s_slide.set_val(s_slide.val + 1)
            else:
                if not (s_slide.val == 0):
                    s_slide.set_val(s_slide.val - 1)

    def MinMaxUpdate(val):
        crt_min = min_slide.val
        crt_max = max_slide.val
        crt_idx = s_slide.val            

        img.set_array(img_array[int(crt_idx),:,:])

        if not(crt_min >= crt_max):
            img.set_clim(vmin=crt_min, vmax=crt_max)
            
        fig.canvas.draw_idle()

    fig = plt.figure(figsize=figsize)
    cid = fig.canvas.mpl_connect('scroll_event', on_press)
    if image_id:
        plt.title(image_id)
        
    # Initial Params
    idx_in = int(img_array.shape[0]/2)
    idx_max = img_array.shape[0]-1

    min_level = min(img_array)
    max_level = max(img_array)

    # Plot image
    extent = [0,img_array.shape[1],0,img_array.shape[2]]
    
    img = plt.imshow(img_array[idx_in,:,:], cmap='gray', extent=extent, aspect='auto', vmin=min_level, vmax=max_level)
    

    # Slice Slider
    s_axes = plt.axes([0.1, 0.06, 0.8, 0.015])
    s_slide = Slider(s_axes, 'slice', 0, idx_max, valinit=idx_in, valfmt='%d', dragging=True, orientation='horizontal',valstep=int(1))
    s_slide.on_changed(MinMaxUpdate)

    # max slicer
    max_axes = plt.axes([0.1, 0.01, 0.8, 0.015])
    max_slide = Slider(max_axes, 'max', 0, max_level, valinit=max_level, valfmt='%d', dragging=True, orientation='horizontal',valstep=int(1))
    max_slide.on_changed(MinMaxUpdate)

    # min slicer
    min_axes = plt.axes([0.1, 0.035, 0.8, 0.015])
    min_slide = Slider(min_axes, 'min', 0, max_level, valinit=min_level, valfmt='%d', dragging=True, orientation='horizontal',valstep=int(1))
    min_slide.on_changed(MinMaxUpdate)
    
    #rectangleSelector
    plt.show()