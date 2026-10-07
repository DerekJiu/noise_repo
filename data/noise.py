import numpy as np
import torch
import matplotlib.pyplot as plt #load file
from PIL import Image
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
  
def add_quantum(image, intensity=0):
    if intensity == 0:
        print("Note: Quantum noise not being applied (intensity = 0)")
        return image
    image_data = np.array(image) / 255.0 
    PHOTON = 1000 # original baseline is at 1000
    photon_count = PHOTON / (intensity**2)
    scaled_image = image_data * photon_count

    poisson_noisy_image = np.random.poisson(scaled_image) / photon_count
    noisy = np.clip(poisson_noisy_image, 0, 1)
    noisy_image = Image.fromarray((noisy * 255).astype(np.uint8))
    return noisy_image


def add_electronic(image, intensity=0):
    if intensity == 0:
        print("Note: Electronic noise not being applied (intensity = 0)")
        return image
    image_data = np.array(image) / 255
    STD_DEV = 0.1 # set baseline at 1%
    sigma = STD_DEV * intensity
    noise = np.random.normal(0, sigma, image_data.shape)
    gaussian_noisy_image = image_data + noise
    noisy = np.clip(gaussian_noisy_image, 0, 1)
    noisy_image = Image.fromarray((noisy * 255).astype(np.uint8))
    return noisy_image


def _show(poisson_noisy_image):
  plt.title('Poisson Noisy X-ray Slice')
  plt.imshow(poisson_noisy_image, cmap='gray')
  plt.axis('off')
  plt.show()
  
if __name__ == "__main__":
  from PIL import Image
  import sys

  # if generating samples, use this format:
  # python noise.py <image_path> <quantum_intensity> <electronic_intensity>
  # assign 0 to either quantum_intensity or electronic_intensity to skip that noise

  try:
    image_path = sys.argv[1]
    quantum_intensity = int(sys.argv[2])
    electronic_intensity = float(sys.argv[3])
  except:
    print("Invalid format: python noise.py <image_path> <quantum_intensity> <electronic_intensity>")
    sys.exit(1)
  image = Image.open(image_path).convert("L") 
  quantum_noisy = add_quantum(image, intensity=quantum_intensity)
  combined_noisy = add_electronic(quantum_noisy, intensity=electronic_intensity)

  #new_add_quantum(image_path, intensity=quantum_intensity)
  
  _show(combined_noisy)
