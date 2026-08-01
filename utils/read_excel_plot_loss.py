import os
import openpyxl
from matplotlib import pyplot as plt


def get_xlsx_filenames_from_folder(input_folder):
    file_list = sorted(os.listdir(input_folder))
    excel_names = []
    for i in file_list:
        if os.path.splitext(i)[1] == ".xlsx":
            excel_names.append(i)
    return excel_names


excel_folder = "..\\modelsave\\EViT_ResUNet3D_AdaptiveGated\\"
excel_names = get_xlsx_filenames_from_folder(excel_folder)

epoch, loss_train, loss_valid = [], [], []
for i in range(len(excel_names)):
    epoch.append(i + 1)
    excel_path = excel_folder + excel_names[i]
    workbook = openpyxl.load_workbook(filename=excel_path)
    sheet = workbook["loss_train_valid_%d" % (i + 1)]
    loss_train.append(sheet.cell(row=1, column=3).value)
    loss_valid.append(sheet.cell(row=1, column=4).value)

plt.figure()
plt.plot(epoch, loss_train, "r", linewidth=2, label="Training Loss")
plt.plot(epoch, loss_valid, color="#328DCA", linewidth=2, label="Testing Loss")
# plt.ylim(0, 130)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="best", frameon=False)
plt.title("Loss of Training and Testing")
plt.savefig("Loss of Training and Testing DATA AdaptiveGated.png")
plt.show()




