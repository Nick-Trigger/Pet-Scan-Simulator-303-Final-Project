
# %%import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageFilter
from scipy.ndimage import label
import matplotlib.pyplot as plt


# %%
# {name : [image_path, whitematter_color, graymatter_color]}
def get_image_info():
    print("{name : [image_path, whitematter_color, graymatter_color]}")
    return {
    'CoronalBrain1' : ['../BrainImages/CoronalBrain1.png', (245, 189, 157), (216, 147, 113)],
    'CoronalBrain2' : ['../BrainImages/CoronalBrain2.png', (250, 202, 182), (211, 153, 119)],
    'CoronalBrain3' : ['../BrainImages/CoronalBrain3.png', (243, 191, 162), (215, 152, 116)],
    'HorizontalBrain1' : ['../BrainImages/HorizontalBrain1.png', (234, 172, 136), (206, 137, 101)],
    'HorizontalBrain2' : ['../BrainImages/HorizontalBrain2.png', (219, 162, 135), (193, 134, 105)],
    'HorizontalBrain3' : ['../BrainImages/HorizontalBrain3.png', (237, 178, 149), (206, 142, 109)]
}

# %%
## Convert image to phantom
def img_to_phantom(img_path, name, whitematter_color, graymatter_color, background_val, whitematter_val, graymatter_val, csf_val, p_whitematter_val, p_graymatter_val, p_csf_val, tumor_params = None, tolerance_pct = 8, brain_bound_padding = 10, blur_radius = 5, dbg = False):
    
    ## -------------------------------------------------
    ## Check if the phantom already exists
    ## -------------------------------------------------
    
    query = f"{name}__{tolerance_pct}_{brain_bound_padding}_{background_val}_{whitematter_val}_{graymatter_val}_{csf_val}_{p_whitematter_val}_{p_graymatter_val}_{p_csf_val}_b_{blur_radius}_t_{tumor_params}"
    
    if os.path.exists(f'../BrainPhantoms/{query}.npy'):
        if not dbg:
            return np.load(f'../BrainPhantoms/{query}.npy'), query
    
    ## -------------------------------------------------
    ## Load the image & remove the number from the top left corner
    ## -------------------------------------------------

    img = Image.open(img_path).convert('RGBA')
    r, g, b, _ = img.split()

    # Remove the number from top left corner of the image
    for x in range(img.width):
        for y in range(img.height): 
            if r.getpixel((x, y)) == 248 and g.getpixel((x, y)) == 248 and b.getpixel((x, y)) == 248:
                img.putpixel((x, y), (0, 0, 0, 0))

    r, g, b, a = img.split()

    ## -------------------------------------------------
    ## Find whitematter and gray matter pixels
    ## -------------------------------------------------
    
    whitematter_img = Image.new("L", img.size, 0)
    graymatter_img = Image.new("L", img.size, 0)
    
    for x in range(img.width):
        for y in range(img.height):
            is_whitematter_r = whitematter_color[0] - tolerance_pct*whitematter_color[0]/100 <= r.getpixel((x, y)) <= whitematter_color[0] + tolerance_pct*whitematter_color[0]/100
            is_whitematter_g = whitematter_color[1] - tolerance_pct*whitematter_color[1]/100 <= g.getpixel((x, y)) <= whitematter_color[1] + tolerance_pct*whitematter_color[1]/100
            is_whitematter_b = whitematter_color[2] - tolerance_pct*whitematter_color[2]/100 <= b.getpixel((x, y)) <= whitematter_color[2] + tolerance_pct*whitematter_color[2]/100
            
            is_graymatter_r = graymatter_color[0] - tolerance_pct*graymatter_color[0]/100 <= r.getpixel((x, y)) <= graymatter_color[0] + tolerance_pct*graymatter_color[0]/100
            is_graymatter_g = graymatter_color[1] - tolerance_pct*graymatter_color[1]/100 <= g.getpixel((x, y)) <= graymatter_color[1] + tolerance_pct*graymatter_color[1]/100
            is_graymatter_b = graymatter_color[2] - tolerance_pct*graymatter_color[2]/100 <= b.getpixel((x, y)) <= graymatter_color[2] + tolerance_pct*graymatter_color[2]/100
            
            if is_whitematter_r and is_whitematter_g and is_whitematter_b:
                whitematter_img.putpixel((x, y), whitematter_val) 
            elif is_graymatter_r and is_graymatter_g and is_graymatter_b:
                graymatter_img.putpixel((x, y), graymatter_val) 
    
    whitematter_array = np.array(whitematter_img)
    graymatter_array = np.array(graymatter_img)
    
    wg_array = np.add(whitematter_array, graymatter_array) # Combine the two arrays
    ## -------------------------------------------------
    ## Set CSF pixels
    ## -------------------------------------------------
    
    rows, cols = np.where(wg_array != 0) # find the size of the brain
    min_x = np.min(cols) - brain_bound_padding
    max_y = np.max(rows) + brain_bound_padding
    max_x = np.max(cols) + brain_bound_padding
    min_y = np.min(rows) - brain_bound_padding
    
    
    # Create a background array and set the oval directly, excluding the brain
    background_array = np.zeros_like(wg_array)
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    width = max_x - min_x
    height = max_y - min_y
    create_oval(background_array, center_x, center_y, width, height, csf_val)
    csf_array = background_array.copy()  # Get the oval
    csf_array[wg_array != 0] = 0  # Ensure no overlap with brain

    if dbg:
        for image, title in zip([csf_array, whitematter_array, graymatter_array, background_array], ['CSF Layer', 'Whitematter Layer', 'Graymatter Layer', 'BKG Layer']):
            plt.imshow(image, cmap='gray')
            plt.title(title)
            plt.show()
            print(f"Max value in {title}: {np.max(image)}")
            print(f"Min value in {title}: {np.min(image)}")
            print(f"Mean value in {title}: {np.mean(image)}")

    ## -------------------------------------------------
    ## add blur to each layer
    ## -------------------------------------------------
    
    whitematter_array = Image.fromarray(whitematter_array).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    graymatter_array = Image.fromarray(graymatter_array).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    csf_array = Image.fromarray(csf_array).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    whitematter_array = np.array(whitematter_array)
    graymatter_array = np.array(graymatter_array)
    csf_array = np.array(csf_array)
    
    ## -------------------------------------------------
    ## Combine the layers
    ## -------------------------------------------------
    
    result = whitematter_array + graymatter_array + csf_array
    result = np.array(result)
    
    if not dbg:
        np.save(f'../BrainPhantoms/{query}.npy', result) # Save the phantom
    
    return result, query


# %%
def create_oval(arr, x, y, w, h, val):
    rows, cols = arr.shape
    y_g, x_g = np.ogrid[-y:rows - y, -x:cols - x]
    mask = (x_g**2 / (w / 2)**2) + (y_g**2 / (h / 2)**2) <= 1
    arr[mask] = val