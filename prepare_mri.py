import os
import nibabel as nib
import matplotlib.pyplot as plt

SOURCE_FOLDER = r"C:\Users\USER\Downloads\ADNI"

SAVE_FOLDER = "dataset/mri_images"

os.makedirs(
    SAVE_FOLDER,
    exist_ok=True
)

count = 0

for root, dirs, files in os.walk(
    SOURCE_FOLDER
):

    for file in files:

        if file.endswith(".nii"):

            try:

                path = os.path.join(
                    root,
                    file
                )

                img = nib.load(
                    path
                )

                data = img.get_fdata()

                middle = (
                    data.shape[2] // 2
                )

                save_path = os.path.join(
                    SAVE_FOLDER,
                    f"scan_{count}.png"
                )

                plt.imsave(
                    save_path,
                    data[:, :, middle],
                    cmap="gray"
                )

                count += 1

                print(
                    "Saved:",
                    count
                )

            except:

                print(
                    "Skipped"
                )

print(
    "DONE"
)