import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_LAB_values(frame_path):
    image = cv2.imread(frame_path)

    # 1. Convert to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 2. Compress image k = 25
    compressed_image = compress_image(lab_image, k=25)

    L = compressed_image[:, :, 0]
    A = compressed_image[:, :, 1]
    B = compressed_image[:, :, 2]

    return L, A, B

def compress_image(uncompressed_image, k=20, plot = False):
    
    # Prepare a compressed image placeholder
    compressed = np.zeros_like(uncompressed_image, dtype=np.float32)
    
    # Apply SVD per channel (compress to k = 25)
    for c in range(3):  # 0=L, 1=A, 2=B
        channel = uncompressed_image[:, :, c].astype(np.float32)
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        
        # Keep top k singular values
        compressed[:, :, c] = (U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :])
    
    if plot:
        # Clip values to valid range
        compressed = np.clip(compressed, 0, 255).astype(np.uint8)
        # Show original and compressed
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.imshow(uncompressed_image)
        plt.title("Original")
        plt.axis("off")
        
        plt.subplot(1,2,2)
        plt.imshow(compressed)
        plt.title(f"Compressed k={k}")
        plt.axis("off")
        
        plt.show()
    return compressed

# ============ Content below here is for personal testing and experimentation - Oliver ============

def test_LAB():
    path = "data/blue/frames1"  # Specify the path (CHANGE THIS)

    # Load an image
    image = cv2.imread(f'{path}/frame455.jpg')
    # Convert BGR → RGB (for display)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert BGR → LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # And compress the image k = 25
    compressed = np.zeros_like(lab_image, dtype=np.float32)
    k = 25
    for c in range(3):
        channel = lab_image[:, :, c]
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        compressed[:, :, c] = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    
    L = compressed[:, :, 0]
    A = compressed[:, :, 1]
    B = compressed[:, :, 2]

    # L = lab_image[:, :, 0]
    # A = lab_image[:, :, 1]
    # B = lab_image[:, :, 2]

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Original Image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original RGB")
    axes[0, 0].axis("off")

    # L channel
    axes[0, 1].imshow(L, cmap='gray')
    axes[0, 1].set_title("L Channel")
    axes[0, 1].axis("off")

    # A channel
    axes[1, 0].imshow(A, cmap='gray')
    axes[1, 0].set_title("A Channel")
    axes[1, 0].axis("off")

    # B channel
    axes[1, 1].imshow(B, cmap='gray')
    axes[1, 1].set_title("B Channel")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    test_LAB()
    # compress_image("data/blue/frames1/frame638.jpg", k=25, plot = True)