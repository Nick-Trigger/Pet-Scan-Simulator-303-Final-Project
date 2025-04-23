# %%import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageFilter
from scipy.ndimage import label
import matplotlib.pyplot as plt
from skimage.transform import rotate, radon, iradon
from tqdm import tqdm

# %%
# {name : [image_path, whitematter_color, graymatter_color]}
# this function gives us all of the information about the images so they can be displayed, it also contains the RBG values of the gray and white matter, which we determined by examining 
# individual pixels in the images
def get_image_info():
    return {
        "CoronalBrain1": [
            "./BrainImages/CoronalBrain1.png",
            (245, 189, 157),
            (216, 147, 113),
        ],
        "CoronalBrain2": [
            "./BrainImages/CoronalBrain2.png",
            (250, 202, 182),
            (211, 153, 119),
        ],
        "CoronalBrain3": [
            "./BrainImages/CoronalBrain3.png",
            (243, 191, 162),
            (215, 152, 116),
        ],
        "HorizontalBrain1": [
            "./BrainImages/HorizontalBrain1.png",
            (234, 172, 136),
            (206, 137, 101),
        ],
        "HorizontalBrain2": [
            "./BrainImages/HorizontalBrain2.png",
            (219, 162, 135),
            (193, 134, 105),
        ],
        "HorizontalBrain3": [
            "./BrainImages/HorizontalBrain3.png",
            (237, 178, 149),
            (206, 142, 109),
        ],
    }


# %%
## Convert image to phantom
def img_to_phantom( # inputs are al of the variable parameters from the GUI, and the image path
    img_path,
    name,
    whitematter_color,
    graymatter_color,
    background_val,
    whitematter_val,
    graymatter_val,
    csf_val,
    p_whitematter_val,
    p_graymatter_val,
    p_csf_val,
    c_skull_val,
    p_skull_val,
    tumor_params=None,
    tolerance_pct=8,
    brain_bound_padding=10,
    blur_radius=5,
    dbg=False,
    fileloc="../BrainPhantoms/",
    skull_thickness=15,
):

    ## Check if the phantom already exists
    query = f"{name}__{tolerance_pct}_{brain_bound_padding}_{background_val}_{whitematter_val}_{graymatter_val}_{csf_val}_{p_whitematter_val}_{p_graymatter_val}_{p_csf_val}_b_{blur_radius}_t_{tumor_params}_s_{skull_thickness}_{c_skull_val}_{p_skull_val}"
    if os.path.exists(f"{fileloc}{query}.npy"):
        if not dbg:
            return np.load(f"{fileloc}{query}.npy"), query

    img = Image.open(img_path).convert("RGBA")
    r, g, b, _ = img.split() # isolates channels

    # Remove the number from top left corner of the image
    for x in range(img.width):
        for y in range(img.height):
            if (
                r.getpixel((x, y)) == 248
                and g.getpixel((x, y)) == 248
                and b.getpixel((x, y)) == 248
            ):
                img.putpixel((x, y), (0, 0, 0, 0))

## -------------------------------------------------
## Pad the y directions (had to do this to eliminate artifacts in the iradonttrasnform, as parts of the phantom were outside the FOV
## -------------------------------------------------
    rows_to_add = img.width - img.height
    
    new_img = Image.new("RGBA", (img.width, img.width), (0, 0, 0, 0))
    new_img.paste(img, (0, rows_to_add // 2))
    img = new_img
        

    r, g, b, a = img.split()


    ## Find whitematter and gray matter pixels

    whitematter_img = Image.new("L", img.size, 0) # initilize empty images to serve as white matter and gray matter
    graymatter_img = Image.new("L", img.size, 0)

    for x in range(img.width):  # note, tolerance percent is set to 8% in function inputs
        for y in range(img.height):
            is_whitematter_r = ( # checks to make sure the pixel is in an acceptable range for white matter's red channel value, the rest work similiarly
                whitematter_color[0] - tolerance_pct * whitematter_color[0] / 100 
                <= r.getpixel((x, y))
                <= whitematter_color[0] + tolerance_pct * whitematter_color[0] / 100
            )
            is_whitematter_g = (
                whitematter_color[1] - tolerance_pct * whitematter_color[1] / 100
                <= g.getpixel((x, y))
                <= whitematter_color[1] + tolerance_pct * whitematter_color[1] / 100
            )
            is_whitematter_b = (
                whitematter_color[2] - tolerance_pct * whitematter_color[2] / 100
                <= b.getpixel((x, y))
                <= whitematter_color[2] + tolerance_pct * whitematter_color[2] / 100
            )

            is_graymatter_r = (
                graymatter_color[0] - tolerance_pct * graymatter_color[0] / 100
                <= r.getpixel((x, y))
                <= graymatter_color[0] + tolerance_pct * graymatter_color[0] / 100
            )
            is_graymatter_g = (
                graymatter_color[1] - tolerance_pct * graymatter_color[1] / 100
                <= g.getpixel((x, y))
                <= graymatter_color[1] + tolerance_pct * graymatter_color[1] / 100
            )
            is_graymatter_b = (
                graymatter_color[2] - tolerance_pct * graymatter_color[2] / 100
                <= b.getpixel((x, y))
                <= graymatter_color[2] + tolerance_pct * graymatter_color[2] / 100
            )

            if is_whitematter_r and is_whitematter_g and is_whitematter_b: # works out if each pixel is white or gray, and adds it to the corresponding blank image
                whitematter_img.putpixel((x, y), 1)
            elif is_graymatter_r and is_graymatter_g and is_graymatter_b:
                graymatter_img.putpixel((x, y), 1)

    whitematter_array = np.array(whitematter_img)
    graymatter_array = np.array(graymatter_img)

    # make copies for PET phantoms
    p_whitematter_array = whitematter_array.copy() * p_whitematter_val # multiplies by whitematter metabolic rate
    p_graymatter_array = graymatter_array.copy() * p_graymatter_val # graymatter metabolic rate

    # assign values to CT phantoms
    whitematter_array = whitematter_array * whitematter_val # white matter density
    graymatter_array = graymatter_array * graymatter_val # gray matter density

    wg_array = np.add(whitematter_array, graymatter_array)  # Combine the two arrays
    
    ## -------------------------------------------------
    ## Set CSF pixels
    ## -------------------------------------------------

    rows, cols = np.where(wg_array != 0)  # find the size of the brain
    min_x = np.min(cols) - brain_bound_padding # brain bound padding set to 10 in inputs
    max_y = np.max(rows) + brain_bound_padding
    max_x = np.max(cols) + brain_bound_padding
    min_y = np.min(rows) - brain_bound_padding

    # CSF ooval
    background_array = np.zeros_like(wg_array)
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    width = max_x - min_x
    height = max_y - min_y
    create_oval(background_array, center_x, center_y, width, height, 1)
    csf_array = background_array.copy()  # Get the oval
    csf_oval = csf_array.copy()  # Make a copy for CSF
    csf_array[wg_array != 0] = 0  # Ensure no overlap with brain
    

    p_csf_array = csf_array.copy()  # Make a copy for PET phantoms
    csf_array = csf_array * csf_val  # Assign CSF value

    if dbg: # development feature for debugging and modification
        for image, title in zip(
            [csf_array, whitematter_array, graymatter_array, background_array],
            ["CSF Layer", "Whitematter Layer", "Graymatter Layer", "BKG Layer"],
        ):
            plt.imshow(image, cmap="gray")
            plt.title(title)
            plt.show()
            print(f"Max value in {title}: {np.max(image)}")
            print(f"Min value in {title}: {np.min(image)}")
            print(f"Mean value in {title}: {np.mean(image)}")

    ## -------------------------------------------------
    ## Add a Skull
    ## -------------------------------------------------
    # Create a mask for the skull
    skull_mask = np.zeros_like(whitematter_array)
    create_oval(
        skull_mask, center_x, center_y, width + skull_thickness, height + skull_thickness, 1
    )
    skull_mask[csf_oval != 0] = 0  # Ensure no overlap with CSF

    # adding gaussian blur to all layers in oder to decrease simulated resolution
    # CT
    skull_mask_c = skull_mask.copy() * c_skull_val
    skull_mask_p = skull_mask.copy() * p_skull_val
    
    whitematter_array = Image.fromarray(whitematter_array.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )
    graymatter_array = Image.fromarray(graymatter_array.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )
    csf_array = Image.fromarray(csf_array.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )
    skull_mask = Image.fromarray(skull_mask_c.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )
    whitematter_array = np.array(whitematter_array)
    graymatter_array = np.array(graymatter_array)
    csf_array = np.array(csf_array)
    skull_mask = np.array(skull_mask)
    
    # PET
    p_whitematter_array = Image.fromarray(p_whitematter_array.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )
    p_graymatter_array = Image.fromarray(p_graymatter_array.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )
    p_csf_array = Image.fromarray(p_csf_array.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )
    p_skull_mask = Image.fromarray(skull_mask_p.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )

    p_whitematter_array = np.array(p_whitematter_array)
    p_graymatter_array = np.array(p_graymatter_array)
    p_csf_array = np.array(p_csf_array)

    ## Add tumor if specified
    if tumor_params is not None:
        x, y, w, h, val, p_val = tumor_params
        tumor_array = np.zeros_like(whitematter_array) #overalay for ct (needs its density value, not metabolic, thats why we need to do these things separeaely)
        create_oval(tumor_array, x, y, w, h, val)

        p_tumor_array = np.zeros_like(p_whitematter_array) # tumor overalay for pet
        create_oval(p_tumor_array, x, y, w, h, p_val)

        tumor_array = Image.fromarray(tumor_array).filter( # add blur same as before
            ImageFilter.GaussianBlur(radius=blur_radius)
        )
        tumor_array = np.array(tumor_array)

        p_tumor_array = Image.fromarray(p_tumor_array).filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        )
        p_tumor_array = np.array(p_tumor_array)


    ## Combine the layers
    result_ct = whitematter_array + graymatter_array + csf_array + skull_mask # adds all arrays together to create the phantom withh all elements
    if tumor_params is not None:
        result_ct += tumor_array
    result_ct = np.array(result_ct)

    result_pet = p_whitematter_array + p_graymatter_array + p_csf_array + p_skull_mask
    if tumor_params is not None:
        result_pet += p_tumor_array
    result_pet = np.array(result_pet)

    result = [result_ct, result_pet] # returns the two phantoms which are assigned to variabbles and used in the main code

    if not dbg:
        np.save(f"{fileloc}{query}.npy", result)  # Save the phantom

    return result, query


# %%
def create_oval(arr, x, y, w, h, val):
    rows, cols = arr.shape
    y_g, x_g = np.ogrid[-y : rows - y, -x : cols - x] # similar function to meshgrid
    mask = (x_g**2 / (w / 2) ** 2) + (y_g**2 / (h / 2) ** 2) <= 1
    arr[mask] = val


def pet_sim(image, decay, fluence, exposure_time):    
    # Generate Poisson-distributed emissions
    decays = np.random.poisson(image * exposure_time * decay)
    
    theta = np.linspace(0., 180., max(image.shape), endpoint=False) # all the code and workflow from 3B
    sinogram = radon(decays, theta=theta)
    
    output = iradon(sinogram, filter_name="hamming", circle=True)
    
    return sinogram, output



""" included the original attempt at pet_sim, which simulated individual emissions as lines across the image (ie a point in l, theta)
def pet_sim(image, emissions, blank):
    blank_sinogram = radon(blank) # make empty sinogram
    for i in range(len(image)):
        for j in range(len(image[0])):
            lines = int(np.round(image[i, j] / image.max() * emissions)) # iterate throuugh the array work out how many emissions a pixel should have
            for k in range(lines):
                angle = np.random.randint(0, 180) 
                theta = np.deg2rad(angle)
                height, width = image.shape
                y = i - height / 2
                x = j - width / 2
                l = int(np.round(x * np.cos(theta) + y * np.sin(theta))) # convert xy values of pixel to L value for given theta
                l_index = l + blank_sinogram.shape[0] // 2
                if 0 <= l_index < blank_sinogram.shape[0]: # add one to the sinogram, indicating that at that l and theta value, a line was detected
                    blank_sinogram[l_index, angle] += 1
    return blank_sinogram 
    # would have then done iradon on this. results were good, however computation time was ridiculous on both our laptops when integrated with main code
    # the above updated version is very efficient, and provides the same results, just not as directly

# some code used for testing original pet_sim
# img_data = pet_sim(np.ones((100, 100)), 1, np.zeros((100, 100)))
# img_data = np.clip(img_data, 0, 255).astype(np.uint8)
# img = Image.fromarray(img_data)
# img2 = iradon(np.array(img))
# max = np.max(img2)
# min = np.min(img2)
# img2 = ((img2 - min) / (max - min)) * 255
# img2 = np.clip(img2, 0, 255).astype(np.uint8)
# img2 = Image.fromarray(img2)
# img2.save("phan.png")
"""
