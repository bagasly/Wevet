import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO


def apply_segmentation(image, segmentation_option, seed_point=None, threshold=None):
    if segmentation_option == "None":
        return image

    if isinstance(image, np.ndarray):  # Pastikan image adalah np.ndarray
        image_array = image
    else:  # Jika image adalah objek PIL.Image, konversi ke np.ndarray
        image_array = np.array(image)

    if len(image_array.shape) > 2:
        image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_array

    segmented_image = np.zeros_like(image_gray)

    if segmentation_option == "Edge Detection":
        edges = cv2.Canny(image_gray, 50, 150)
        segmented_image = edges
    elif segmentation_option == "Region Growing":
        if seed_point is not None and threshold is not None:
            segmented_image = region_growing_segmentation(
                image_array, seed_point, threshold)

    return segmented_image


def region_growing_segmentation(image, seed_point, threshold):
    if len(image.shape) > 2:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image

    segmented_image = np.zeros_like(image_gray, dtype=np.uint8)
    visited = np.zeros_like(image_gray, dtype=np.uint8)

    def grow_region(x, y):
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if visited[y, x] == 1:
                continue
            visited[y, x] = 1

            neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < image_gray.shape[1] and 0 <= ny < image_gray.shape[0]:
                    if segmented_image[ny, nx] == 0 and abs(int(image_gray[ny, nx]) - int(image_gray[y, x])) < threshold:
                        segmented_image[ny, nx] = 255
                        stack.append((nx, ny))

    # Swap x and y for OpenCV indexing
    seed_point = (seed_point[1], seed_point[0])
    if 0 <= seed_point[0] < image_gray.shape[1] and 0 <= seed_point[1] < image_gray.shape[0]:
        segmented_image[seed_point[1], seed_point[0]] = 255
        grow_region(seed_point[1], seed_point[0])

    return segmented_image


def apply_filter(image, filter_option):
    if filter_option == "Grayscale":
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    elif filter_option == "HSV":
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    elif filter_option == "Highpass":
        # Implement Highpass filter
        highpass_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(np.array(image), -1, highpass_kernel)
    elif filter_option == "Lowpass":
        # Implement Lowpass filter (GaussianBlur as an example)
        return cv2.GaussianBlur(np.array(image), (15, 15), 0)
    elif filter_option == "Gaussian":
        # Implement Gaussian filter
        return cv2.GaussianBlur(np.array(image), (5, 5), 0)
    elif filter_option == "Emboss":
        # Implement Emboss filter
        emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        return cv2.filter2D(np.array(image), -1, emboss_kernel)
    elif filter_option == "Inverse":
        # Implement Inverse filter
        return cv2.bitwise_not(np.array(image))
    elif filter_option == "LAB":
        # Convert to LAB color space
        lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
        # Extract the L channel
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        return cv2.merge([l_channel, a_channel, b_channel])
    else:
        return np.array(image)


st.markdown("<h1 style='text-align: center;'> > Image Editor < </h1>",
            unsafe_allow_html=True)
st.markdown("---")

# Load Image
image_path = st.sidebar.file_uploader(
    "Upload Image", type=["jpg", "png", "jpeg"])

info = st.empty()
size = st.empty()
mode = st.empty()
format_ = st.empty()

if image_path:
    original_image = Image.open(image_path)
    st.markdown("<h5 style='text-align: center;'> Original Image </h5>",
                unsafe_allow_html=True)
    st.image(original_image, use_column_width=False)

    info.markdown("<h2 style='text-align:center;'> Information </h2>",
                  unsafe_allow_html=True)
    size.markdown(
        f"<h6> Size: {original_image.size} </h6>", unsafe_allow_html=True)
    mode.markdown(
        f"<h6> Mode: {original_image.mode} </h6>", unsafe_allow_html=True)
    format_.markdown(
        f"<h6> Format: {original_image.format} </h6>", unsafe_allow_html=True)

    # Sidebar Options
    st.sidebar.title("Edit Options")

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    # Resize
    new_size = st.sidebar.slider("Resize", 10, 1000, 300)
    resized_image = original_image.resize((new_size, new_size))

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    # Rotate
    rotation_angle = st.sidebar.slider("Rotate", -180, 180, 0)
    rotated_image = resized_image.rotate(rotation_angle)

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    # Image Filter
    filter_option = st.sidebar.selectbox("Filter Options", [
                                         "None", "Grayscale", "HSV", "Highpass", "Lowpass", "Gaussian", "Emboss", "Inverse", "LAB"])

    if filter_option != "None":
        edited_image = apply_filter(rotated_image, filter_option)
        # Display Edited Image
        st.markdown("---")
        st.markdown("<h5 style='text-align: center;'> Edited Image </h5>",
                    unsafe_allow_html=True)
        st.image(
            edited_image, caption=f"Filtered Image - {filter_option}", use_column_width=False)
    else:
        # Display Edited Image
        st.markdown("---")
        st.markdown("<h5 style='text-align: center;'> Edited Image </h5>",
                    unsafe_allow_html=True)
        st.image(rotated_image, caption="Original Image",
                 use_column_width=False)

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    # Select segmentation method
    segmentation_option = st.sidebar.selectbox("Select Segmentation Method", [
                                               "None", "Edge Detection", "Region Growing"])

    if segmentation_option == "Region Growing":
        seed_point = st.sidebar.text_input("Seed Point (x, y)", "50, 50")
        threshold = st.sidebar.slider("Threshold", 1, 255, 20)
        seed_point = tuple(map(int, seed_point.split(',')))

        # Perform region growing segmentation
        segmented_image = apply_segmentation(
            rotated_image, segmentation_option, seed_point=seed_point, threshold=threshold)
    else:
        # Perform other segmentations
        segmented_image = apply_segmentation(
            rotated_image, segmentation_option)

    # Display segmented image
    st.image(segmented_image,
             caption=f"Segmented Image - {segmentation_option}", use_column_width=False)

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    # Show Histogram Button
    if st.sidebar.button("Show Histogram"):
        # Calculate histogram
        hist, bins = np.histogram(
            np.array(segmented_image).flatten(), 256, [0, 256])

        # Plot histogram
        fig, ax = plt.subplots()
        ax.plot(hist)
        ax.set_title("Image Histogram")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")

        # Display plot in Streamlit
        st.markdown("---")
        st.markdown("<h5 style='text-align: center;'> Histogram Table </h5>",
                    unsafe_allow_html=True)
        st.pyplot(fig)

        # Save Histogram Data as CSV
        histogram_data = pd.DataFrame(
            {"Pixel Value": bins[:-1], "Frequency": hist})
        st.sidebar.download_button(
            label="Download Histogram Data",
            data=histogram_data.to_csv(index=False),
            file_name="histogram_data.csv",
            key="download_histogram",
        )

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # Save As
    save_format = st.sidebar.selectbox("Save As", ["JPG", "PNG", "PDF"])
    if st.sidebar.button("Download Edited Image"):
        # Convert edited image to bytes
        image_bytes = BytesIO()

        if segmentation_option == "None":
            if filter_option == "None":
                edited_image_array = np.array(rotated_image)
            else:
                edited_image_array = np.array(edited_image)
        else:
            edited_image_array = np.array(segmented_image)

        if save_format.upper() == "JPG":
            edited_image_pil = Image.fromarray(
                edited_image_array.astype('uint8'))
            edited_image_pil.save(image_bytes, format="JPEG")
        else:
            edited_image_pil = Image.fromarray(edited_image_array)
            edited_image_pil.save(image_bytes, format=save_format.upper())

        # Download button
        st.sidebar.download_button(
            label=f"Download Edited Image ({save_format})",
            data=image_bytes.getvalue(),
            file_name=f"edited_image.{save_format.lower()}",
            key="download_edited_image",
        )
