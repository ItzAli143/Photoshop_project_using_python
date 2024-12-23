import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageOps, ImageFilter
import numpy as np
import cv2
import os

# Initialize global variables
current_image = None
current_file_path = None
zoom_slider = None
zoom_label = None
zoom_percentage = 100
filter_slider = None
filter_label = None
filtered_image = None
rgb_buttons = []  # List to track RGB buttons
current_filter_type = None

# Function to display the zoomed or filtered image
def display_zoomed_image(image=None):
    for widget in center_frame.winfo_children():
        widget.destroy()
    if image is None:
        image = current_image
    if image:
        # Apply zoom to the selected image
        width, height = image.size
        zoomed_width = int(width * (zoom_percentage / 100))
        zoomed_height = int(height * (zoom_percentage / 100))
        zoomed_image = image.resize((zoomed_width, zoomed_height))
        tk_image = ImageTk.PhotoImage(zoomed_image)
        image_label = tk.Label(center_frame, image=tk_image, bg="white")
        image_label.image = tk_image
        image_label.pack(expand=True)

# Function to handle "New"
def file_new():
    global current_image, current_file_path, zoom_percentage, filtered_image
    current_image = None
    current_file_path = None
    zoom_percentage = 100
    filtered_image = None
    display_zoomed_image()  # Clears the center area
    info_text.config(text="New file created. Ready to load an image.")
    hide_zoom_controls()
    hide_filter_controls()

# Function to handle "Open"
def file_open():
    global current_image, current_file_path, zoom_percentage, filtered_image
    file_path = filedialog.askopenfilename(
        title="Open Image",
        filetypes=(("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), ("All Files", "*.*"))
    )
    if file_path:
        current_file_path = file_path
        pil_image = Image.open(file_path)
        pil_image.thumbnail((800, 600))  # Resize for display
        current_image = pil_image
        filtered_image = None  # Reset filtered image
        zoom_percentage = 100  # Reset zoom level to default
        display_zoomed_image()  # Display the image

        file_size = os.path.getsize(file_path) / 1024  # File size in KB
        info_text.config(
            text=f"File Name: {os.path.basename(file_path)} | Size: {file_size:.2f} KB | Dimensions: {pil_image.width}x{pil_image.height}"
        )

        # Show zoom slider
        create_zoom_controls()

# Function to handle "Save"
def file_save():
    global current_image, current_file_path, filtered_image
    if filtered_image:  # Save the filtered image if it exists
        image_to_save = filtered_image
    else:
        image_to_save = current_image  # Otherwise, save the original image

    if image_to_save and current_file_path:
        try:
            image_to_save.save(current_file_path)  # Save the image
            messagebox.showinfo("Save File", f"File saved successfully at {current_file_path}")
        except Exception as e:
            messagebox.showerror("Save File", f"Error saving file: {e}")
    else:
        messagebox.showwarning("Save File", "No file to save. Use 'Save As' instead.")

# Function to handle "Save As"
def file_save_as():
    global current_image, current_file_path, filtered_image
    if filtered_image:  # Save the filtered image if it exists
        image_to_save = filtered_image
    else:
        image_to_save = current_image  # Otherwise, save the original image

    if image_to_save:
        file_path = filedialog.asksaveasfilename(
            title="Save As",
            defaultextension=".png",
            filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg;*.jpeg"), ("BMP Files", "*.bmp"), ("All Files", "*.*"))
        )
        if file_path:
            try:
                image_to_save.save(file_path)  # Save the image
                current_file_path = file_path  # Update the current file path
                messagebox.showinfo("Save As", f"File saved successfully at {file_path}")
            except Exception as e:
                messagebox.showerror("Save As", f"Error saving file: {e}")
    else:
        messagebox.showwarning("Save As", "No file to save. Please open or create a file first.")

# Function to handle "Exit"
def file_exit():
    if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
        root.destroy()

# Functionality for Zoom slider
def update_zoom(event=None):
    global zoom_percentage
    if zoom_slider:
        zoom_percentage = zoom_slider.get()
        display_zoomed_image(filtered_image if filtered_image else current_image)
        if zoom_label:  # Check if zoom_label exists before updating it
            zoom_label.config(text=f"Zoom: {zoom_percentage:.0f}%")

# Create Zoom Controls
def create_zoom_controls():
    global zoom_slider, zoom_label
    if not zoom_slider:  # Only create controls if they don't already exist
        zoom_slider = ttk.Scale(info_bar, from_=10, to=200, orient="horizontal", command=update_zoom)
        zoom_slider.set(100)
        zoom_slider.pack(side="right", padx=10, pady=5)

        zoom_label = tk.Label(info_bar, text="Zoom: 100%", bg="orange", fg="purple", font=("Arial", 10))
        zoom_label.pack(side="right", padx=10)

# Hide Zoom Controls
def hide_zoom_controls():
    global zoom_slider, zoom_label
    if zoom_slider:
        zoom_slider.destroy()
        zoom_slider = None
    if zoom_label:
        zoom_label.destroy()
        zoom_label = None
# Hide Filter Controls
def hide_filter_controls():
    global filter_slider, filter_label, rgb_buttons
    if filter_slider:
        filter_slider.destroy()
        filter_slider = None
    if filter_label:
        filter_label.destroy()
        filter_label = None
    for button in rgb_buttons:
        button.destroy()
    rgb_buttons.clear()

# Functionality for Mean Filter
def apply_mean_filter(kernel_size):
    global filtered_image
    if current_image:
        # Convert image to numpy array for filtering
        img_array = np.array(current_image)
        # Apply mean filter (simple box filter)
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        filtered_array = np.zeros_like(img_array)

        # Apply convolution to each channel
        for c in range(3):  # Assuming RGB image
            filtered_array[:, :, c] = np.convolve(
                img_array[:, :, c].flatten(), kernel.flatten(), mode='same'
            ).reshape(img_array.shape[:2])

        filtered_image = Image.fromarray(filtered_array)  # Update the global filtered_image
        display_zoomed_image(filtered_image)  # Show the filtered image

# Create Mean Filter Controls
def create_filter_controls():
    global filter_slider, filter_label
    hide_filter_controls()  # Remove any existing controls
    filter_label = tk.Label(right_frame, text="Kernel Size: 3", bg="orange", fg="purple", font=("Arial", 12))
    filter_label.pack(padx=10, pady=5)
    filter_slider = ttk.Scale(right_frame, from_=3, to=15, orient="horizontal", command=update_filter, length=150)
    filter_slider.set(3)  # Default kernel size
    filter_slider.pack(padx=10, pady=5)

# Update Mean Filter dynamically
def update_filter(event=None):
    if filter_slider:
        kernel_size = int(filter_slider.get())
        filter_label.config(text=f"Kernel Size: {kernel_size}")
        apply_mean_filter(kernel_size)

# Functionality for Gaussian Blur Filter
def apply_gaussian_blur(radius):
    global filtered_image
    if current_image:
        # Apply Gaussian Blur with the specified radius
        filtered_image = current_image.filter(ImageFilter.GaussianBlur(radius=radius))
        display_zoomed_image(filtered_image)  # Display the blurred image

# Create Gaussian Blur Filter Controls
# Create Gaussian Blur Filter Controls (continued)
def create_gaussian_blur_controls():
    global filter_slider, filter_label
    hide_filter_controls()  # Remove any existing controls to avoid conflicts
    filter_label = tk.Label(right_frame, text="Blur Radius: 1.0", bg="orange", fg="purple", font=("Arial", 12))
    filter_label.pack(padx=10, pady=5)
    filter_slider = ttk.Scale(right_frame, from_=0.1, to = 10, orient="horizontal", command=update_gaussian_blur, length=150)
    filter_slider.set(1.0)  # Default blur radius
    filter_slider.pack(padx=10, pady=5)
def update_gaussian_blur(event=None):
    if filter_slider:
        radius = float(filter_slider.get())
        filter_label.config(text=f"Blur Radius: {radius:.1f}")
        apply_gaussian_blur(radius)
# Functionality for Laplacian of Gaussian Filter
def apply_log_filter(sigma):
    global filtered_image
    if current_image:
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(current_image), cv2.COLOR_RGB2BGR)

        # Apply LoG filter
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = np.dot(kernel, kernel.T)
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        kernel = cv2.filter2D(kernel, -1, laplacian_kernel)

        filtered_img_cv = cv2.filter2D(img_cv, -1, kernel)

        # Convert back to PIL Image
        filtered_image = Image.fromarray(cv2.cvtColor(filtered_img_cv, cv2.COLOR_BGR2RGB))

        display_zoomed_image(filtered_image)

# Create LoG Filter Controls
def create_log_filter_controls():
    global filter_slider, filter_label
    hide_filter_controls()  # Remove any existing controls to avoid conflicts
    filter_label = tk.Label(right_frame, text="Sigma: 1.0", bg="orange", fg="purple", font=("Arial", 12))
    filter_label.pack(padx=10, pady=5)
    filter_slider = ttk.Scale(right_frame, from_=0.1, to=5, orient="horizontal", command=update_log_filter, length=150)
    filter_slider.set(1.0)  # Default sigma value
    filter_slider.pack(padx=10, pady=5)

# Update LoG Filter dynamically
def update_log_filter(event=None):
    if filter_slider:
        sigma = filter_slider.get()
        filter_label.config(text=f"Sigma: {sigma:.1f}")
        apply_log_filter(sigma)

# Functionality for Grayscale Filter
def apply_grayscale_filter():
    global filtered_image
    if current_image:
        filtered_image = ImageOps.grayscale(current_image)
        display_zoomed_image(filtered_image)
        update_sliders("grayscale")  # Hide any existing sliders

# Create Grayscale Filter Controls
def create_grayscale_filter_controls():
    apply_grayscale_filter()

# Function to update sliders based on filter type
def update_sliders(filter_type):
    global filter_slider, filter_label
    if filter_type == "grayscale":
        # Check if filter_slider exists before hiding it
        if filter_slider:
            filter_slider.pack_forget()
        if filter_label:
            filter_label.pack_forget()
    else:
        # Show the filter slider and label for other filters
        if filter_slider:
            filter_slider.pack()
        if filter_label:
            filter_label.pack()

# Functionality to display different channels
def apply_rgb_channel(channel):
    global filtered_image
    if current_image:
        # Convert the image to NumPy array
        img_array = np.array(current_image)
        
        # Split the channels
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        # Create a blank array for the channels
        blank = np.zeros_like(r)

        # Select the requested channel
        if channel == "red":
            channel_array = np.stack((r, blank, blank), axis=-1)
        elif channel == "green":
            channel_array = np.stack((blank, g, blank), axis=-1)
        elif channel == "blue":
            channel_array = np.stack((blank, blank, b), axis=-1)
        elif channel == "original":
            channel_array = img_array  # Original image

        # Convert back to PIL Image
        filtered_image = Image.fromarray(channel_array)
        display_zoomed_image(filtered_image)

def create_rgb_channel_controls():
    global rgb_buttons
    hide_filter_controls()
    original_button = tk.Button(right_frame, text="Original", bg="white", fg="black", command=lambda: apply_rgb_channel("original"))
    original_button.pack(padx=10, pady=5)
    rgb_buttons.append(original_button)
    red_button = tk.Button(right_frame, text="Red Channel", bg="red", fg="white", command=lambda: apply_rgb_channel("red"))
    red_button.pack(padx=10, pady=5)
    rgb_buttons.append(red_button)
    green_button = tk.Button(right_frame, text="Green Channel", bg="green", fg="white", command=lambda: apply_rgb_channel("green"))
    green_button.pack(padx=10, pady=5)
    rgb_buttons.append(green_button)
    blue_button = tk.Button(right_frame, text="Blue Channel", bg="blue", fg="white", command=lambda: apply_rgb_channel("blue"))
    blue_button.pack(padx=10, pady=5)
    rgb_buttons.append(blue_button)

def create_filters_menu():
    filters_menu = tk.Menu(menu_bar, tearoff=0)
    filters_menu.add_command(label="Mean Filter", command=lambda: [create_filter_controls(), apply_mean_filter(3)])
    filters_menu.add_command(label="Gaussian Blur", command=lambda: [create_gaussian_blur_controls(), apply_gaussian_blur(1)])
    filters_menu.add_command(label="Laplacian", command=lambda: [create_log_filter_controls(), apply_log_filter(1)])
    filters_menu.add_command(label="Grayscale", command=create_grayscale_filter_controls)
    filters_menu.add_command(label="RGB Channels", command=create_rgb_channel_controls)  # New Option
    menu_bar.add_cascade(label="Filters", menu=filters_menu)
# Function to rotate image
def rotate_image():
    global current_image, filtered_image
    hide_filter_controls()  # Hide any existing sliders
    if current_image:
        # If there's already a rotated image, rotate it further
        image_to_rotate = filtered_image if filtered_image else current_image
        rotated_image = image_to_rotate.rotate(-90, expand=True)  # Rotate 90 degrees clockwise
        filtered_image = rotated_image
        display_zoomed_image(filtered_image)

# Function to undo changes and display the original image
def undo_changes():
    global filtered_image
    hide_filter_controls()  # Hide any existing sliders
    filtered_image = None  # Reset filtered image
    display_zoomed_image(current_image)  # Display the original image

# Functionality to adjust brightness
def apply_brightness(factor):
    global current_image, filtered_image
    if current_image:
        enhancer = ImageEnhance.Brightness(current_image)
        brightened_image = enhancer.enhance(factor)
        filtered_image = brightened_image
        display_zoomed_image(filtered_image)

# Create Brightness Control Slider
def create_brightness_control():
    global filter_slider, filter_label
    hide_filter_controls()  # Hide any existing controls
    filter_label = tk.Label(right_frame, text="Brightness: 1.0", bg="orange", fg="purple", font=("Arial", 12))
    filter_label.pack(padx=10, pady=5)
    filter_slider = ttk.Scale(right_frame, from_=0.5, to=2.0, orient="horizontal", command=update_brightness, length=150)
    filter_slider.set(1.0)  # Default brightness
    filter_slider.pack(padx=10, pady=5)

# Function to update brightness dynamically
def update_brightness(event=None):
    if filter_slider:
        factor = filter_slider.get()
        filter_label.config(text=f"Brightness: {factor:.1f}")
        apply_brightness(factor)

# Create new effects menu
def create_effects_menu():
    effects_menu = tk.Menu(menu_bar, tearoff=0)
    effects_menu.add_command(label="Edge Detection", command=apply_edge_detection)
    effects_menu.add_command(label="Invert Colors", command=apply_invert_colors)
    effects_menu.add_command(label="Corner Detection", command=apply_corner_detection)  # Existing Option
    
    # Create a new submenu for Image Derivatives
    derivatives_menu = tk.Menu(effects_menu, tearoff=0)
    derivatives_menu.add_command(label="First Order Derivative", command=apply_first_order_derivative)
    derivatives_menu.add_command(label="Second Order Derivative", command=apply_second_order_derivative)
    
    # Add Image Derivatives submenu to Special Tools menu
    effects_menu.add_cascade(label="Image Derivatives", menu=derivatives_menu)
    
    menu_bar.add_cascade(label="Special Tools", menu=effects_menu)

# Function to apply edge detection
# Function to apply edge detection (continued)
def apply_edge_detection():
    global filtered_image
    hide_filter_controls()  # Hide any existing sliders
    if current_image:
        img_cv = cv2.cvtColor(np.array(current_image), cv2.COLOR_RGB2BGR)
        edges = cv2.Canny(img_cv, 100, 200)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        filtered_image = Image.fromarray(edges_rgb)
        display_zoomed_image(filtered_image)

# Function to invert colors
def apply_invert_colors():
    global filtered_image
    hide_filter_controls()  # Hide any existing sliders
    if current_image:
        inverted_image = ImageOps.invert(current_image.convert("RGB"))
        filtered_image = inverted_image
        display_zoomed_image(filtered_image)

# Function to apply first order derivative
def apply_first_order_derivative():
    global filtered_image
    hide_filter_controls()  # Hide any existing sliders
    if current_image:
        img_cv = cv2.cvtColor(np.array(current_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Apply Sobel filter for first order derivative
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)

        # Convert to RGB for display
        sobel_combined_rgb = cv2.cvtColor(np.uint8(sobel_combined), cv2.COLOR_GRAY2RGB)
        filtered_image = Image.fromarray(sobel_combined_rgb)
        display_zoomed_image(filtered_image)

# Function to apply second order derivative
def apply_second_order_derivative():
    global filtered_image
    hide_filter_controls()  # Hide any existing sliders
    if current_image:
        img_cv = cv2.cvtColor(np.array(current_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Apply Laplacian filter for second order derivative
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Convert to RGB for display
        laplacian_rgb = cv2.cvtColor(np.uint8(np.absolute(laplacian)), cv2.COLOR_GRAY2RGB)
        filtered_image = Image.fromarray(laplacian_rgb)
        display_zoomed_image(filtered_image)

# Function to apply corner detection
def apply_corner_detection():
    global filtered_image
    hide_filter_controls()  # Hide any existing sliders
    if current_image:
        img_cv = cv2.cvtColor(np.array(current_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Harris Corner Detection
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        # Result is dilated for marking the corners
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        img_cv[dst > 0.01 * dst.max()] = [0, 0, 255]

        # Convert back to RGB for display
        filtered_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        display_zoomed_image(filtered_image)
#============================================================================================================================
#============================================================================================================================
#============================================================================================================================
#Theme section

def set_dark_theme():
    root.configure(bg="black")
    left_frame.configure(bg="gray25")
    center_frame.configure(bg="black")
    right_frame.configure(bg="gray25")
    info_bar.configure(bg="gray25")
    info_text.configure(bg="gray25", fg="white")
    if zoom_label:
        zoom_label.configure(bg="gray25", fg="white")
    for button in rgb_buttons:
        button.configure(bg="gray25", fg="white")
    if filter_label:
        filter_label.configure(bg="gray25", fg="white")
    # Update the image label background if it exists
    for widget in center_frame.winfo_children():
        if isinstance(widget, tk.Label):
            widget.configure(bg="black")
def set_white_theme():
    root.configure(bg="white")
    left_frame.configure(bg="lightgray")
    center_frame.configure(bg="white")
    right_frame.configure(bg="lightgray")
    info_bar.configure(bg="orange")
    info_text.configure(bg="orange", fg="purple")
    if zoom_label:
        zoom_label.configure(bg="orange", fg="purple")
    for button in rgb_buttons:
        button.configure(bg="lightgray", fg="black")
    if filter_label:
        filter_label.configure(bg="orange", fg="purple")
    # Update the image label background if it exists
    for widget in center_frame.winfo_children():
        if isinstance(widget, tk.Label):
            widget.configure(bg="white")

def create_theme_menu():
    theme_menu = tk.Menu(menu_bar, tearoff=0)
    theme_menu.add_command(label="Dark", command=set_dark_theme)
    theme_menu.add_command(label="White", command=set_white_theme)
    menu_bar.add_cascade(label="Theme", menu=theme_menu)

#============================================================================================================================
#============================================================================================================================
#============================================================================================================================
# about section
def about_app():
    description = """
    This is an Enhanced Computer Vision Application developed to provide various image processing and editing tools.
    Key Features:
    - File Handling: Open, Save, and Save As images.
    - Zoom: Zoom in and out of images.
    - Filters: Apply Mean Filter, Gaussian Blur, Laplacian of Gaussian, Grayscale, and RGB Channel filters.
    - Special Tools: Edge Detection, Color Inversion, Corner Detection, and Image Derivatives.
    - Rotation: Rotate images 90 degrees clockwise.
    - Brightness Adjustment: Increase or decrease image brightness with a slider.
    - Themes: Switch between Dark and White themes.
    """
    display_text(description)

def display_text(text):
    for widget in center_frame.winfo_children():
        widget.destroy()
    text_label = tk.Label(center_frame, text=text, bg="white", justify="left", anchor="nw", font=("Arial", 12))
    text_label.pack(fill="both", expand=True, padx=10, pady=10)

def about_developers():
    for widget in center_frame.winfo_children():
        widget.destroy()

    center_inner_frame = tk.Frame(center_frame, bg="white")
    center_inner_frame.place(relx=0.5, rely=0.5, anchor="center")

    # Load and display Azkeel Asim's image
    azkeel_image = Image.open(r"icons\azkeelphoto.jpg")  # Update with the actual path
    azkeel_image = azkeel_image.resize((250, 250), Image.LANCZOS)  # Resize image
    azkeel_photo = ImageTk.PhotoImage(azkeel_image)
    azkeel_label = tk.Label(center_inner_frame, image=azkeel_photo, text="Azkeel Asim\n(Frontend Developer)", compound="top", font=("Arial", 12))
    azkeel_label.image = azkeel_photo  # Keep a reference to avoid garbage collection
    azkeel_label.pack(side="left", padx=10, pady=10)

    # Load and display Mustansar Ali's image
    mustansar_image = Image.open(r"icons/aliphoto.jpg")  # Update with the actual path
    mustansar_image = mustansar_image.resize((250, 250), Image.LANCZOS)  # Resize image
    mustansar_photo = ImageTk.PhotoImage(mustansar_image)
    mustansar_label = tk.Label(center_inner_frame, image=mustansar_photo, text="Mustansar Ali\n(Backend Developer)", compound="top", font=("Arial", 12))
    mustansar_label.image = mustansar_photo  # Keep a reference to avoid garbage collection
    mustansar_label.pack(side="left", padx=10, pady=10)



def create_about_menu():
    about_menu = tk.Menu(menu_bar, tearoff=0)
    about_menu.add_command(label="About App", command=about_app)
    about_menu.add_command(label="About Developers", command=about_developers)
    menu_bar.add_cascade(label="About", menu=about_menu)


#============================================================================================================================
#============================================================================================================================
#============================================================================================================================
#============================================================================================================================
#============================================================================================================================

# Create custom menu
def create_custom_menu():
    global menu_bar
    menu_bar = tk.Menu(root)
    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="New", command=file_new)
    file_menu.add_command(label="Open", command=file_open)
    file_menu.add_command(label="Save", command=file_save)
    file_menu.add_command(label="Save As", command=file_save_as)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=file_exit)
    menu_bar.add_cascade(label="File", menu=file_menu)

    create_filters_menu()
    create_effects_menu()
    create_theme_menu()
    create_about_menu()
    root.config(menu=menu_bar)



# Initialize the Tkinter root window
root = tk.Tk()
root.title("HD FILTERS")
root.iconbitmap(r"icons\logoo.ico")  # Replace with the path to your .ico file
root.state('zoomed')  # Open in full screen

# Define the layout with three main sections
root.columnconfigure(1, weight=1)
root.rowconfigure(1, weight=1)

left_frame = tk.Frame(root, bg="lightgray", width=150)
left_frame.grid(row=1, column=0, sticky="ns")

center_frame = tk.Frame(root, bg="white")
center_frame.grid(row=1, column=1, sticky="nsew")

right_frame = tk.Frame(root, bg="lightgray", width=200)
right_frame.grid(row=1, column=2, sticky="ns")

info_bar = tk.Frame(root, bg="orange")
info_bar.grid(row=2, column=0, columnspan=3, sticky="ew")

info_text = tk.Label(info_bar, text="Ⓒ 2024 Copyright by Hajvery Dauran Technologies All rights reserved.",
                     bg="orange", fg="purple", anchor="w", font=("Arial", 10))
info_text.pack(side="left", padx=10)

image_canvas = tk.Canvas(center_frame, bg="white")
image_canvas.pack(fill="both", expand=True)
#============================================================================================================================
#============================================================================================================================
#============================================================================================================================

# Function to load icons
def load_icon(icon_path, size=(40, 40)):
    icon = Image.open(icon_path)
    icon = icon.resize(size, Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
    return ImageTk.PhotoImage(icon)

# Load the icons
icon_rotate = load_icon(r"icons\rotate.png")
icon_undo = load_icon(r"icons\undo.png")
icon_brightness = load_icon(r"icons\brightness.png")

# Function to create icon buttons in the left frame
def create_icon_buttons():
    rotate_button = tk.Button(left_frame, image=icon_rotate, command=rotate_image)
    rotate_button.image = icon_rotate  # Keep a reference to avoid garbage collection
    rotate_button.pack(pady=5)

    undo_button = tk.Button(left_frame, image=icon_undo, command=undo_changes)
    undo_button.image = icon_undo
    undo_button.pack(pady=5)

    brightness_button = tk.Button(left_frame, image=icon_brightness, command=create_brightness_control)
    brightness_button.image = icon_brightness 
    brightness_button.pack(pady=5)

# Call the function to create icon buttons
create_icon_buttons()
create_custom_menu()

#============================================================================================================================

root.mainloop()
