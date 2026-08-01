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


excel_folder = "..\\modelsave\\ResUnet3D\\"
excel_names = get_xlsx_filenames_from_folder(excel_folder)

epoch, dice_train, dice_valid = [], [], []
for i in range(len(excel_names)):
    epoch.append(i + 1)
    excel_path = excel_folder + excel_names[i]
    workbook = openpyxl.load_workbook(filename=excel_path)
    sheet = workbook["loss_train_valid_%d" % (i + 1)]
    dice_train.append(sheet.cell(row=1, column=5).value)
    dice_valid.append(sheet.cell(row=1, column=6).value)

plt.figure()
plt.plot(epoch, dice_train, "r", linewidth=2, label="Training DSC")
plt.plot(epoch, dice_valid, color="#328DCA", linewidth=2, label="Testing DSC")
# plt.ylim(0, 130)
plt.xlabel("Epoch")
plt.ylabel("DSC")
plt.legend(loc="best", frameon=False)
plt.title("DSC of Training and Testing")
plt.savefig("DSC of Training and Testing DATA AdaptiveGated_Double.png")
plt.show()




