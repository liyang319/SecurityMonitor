import argparse
import pointer_meter_match
import importlib
from enum import Enum


class MonitorType(Enum):
    POINTER_METER = 'pointer_meter'
    LIGHT_METER = 'light_meter'



def find_templateclass_using_name(class_type, class_name):
    typeHeader = 'template_process.'
    if class_type == MonitorType.POINTER_METER:
        typeHeader += str(MonitorType.POINTER_METER.value)
    elif class_type == MonitorType.LIGHT_METER:
        typeHeader += str(MonitorType.LIGHT_METER.value)
    templateclass_name = typeHeader + '.' + str(class_name)
    print(templateclass_name)
    templateclass = importlib.import_module(templateclass_name)

    if templateclass is None:
        raise NotImplementedError("In templater_process package, the model %s not find." % (templateclass_name))

    return templateclass


if __name__ == "__main__":
    queryImagePath = "./img_new/img00.png"  # the image to be corrected
    templateImgDir = "./template_img/pointer_meter/"  # the tamplate dir
    outImg = "./img_test_corrected/"
    matchedTemplateClass = pointer_meter_match.CorrectImage(queryImagePath, templateImgDir, outImg)
    templateclass = find_templateclass_using_name(MonitorType.POINTER_METER, matchedTemplateClass)
    templateclass.testFun()
    print('------\n')

    # matchedTemplateClass = img_match.CorrectImage(queryImagePath, templateImgDir, outImg)
    # print(matchedTemplateClass)
    #
    # # check the pointer position and compute the num according to the degree of pointer
    # if matchedTemplateClass is None:
    #     raise ValueError("no find the right template class")
    #
    # corrected_img_path = outImg + queryImagePath.split("/")[-1]
    # templateclass = find_templateclass_using_name(matchedTemplateClass)
    #
    # num = templateclass.degree2num(corrected_img_path)
    # print(num)







# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--img_path", type=str, default="./img_test/test1.png",
#                     help='the path of the test image')
# parser.add_argument("--template_dir", type=str, default="./template/",
#                     help='the dir of template images')
# parser.add_argument("--siftedimg_dir", type=str, default="./img_test_corrected/",
#                     help='the dir of sifted images')
#
# opt, _ = parser.parse_known_args()
#
# # find the right template and correct the image
# queryImagePath = opt.img_path
# templateImgDir = opt.template_dir
# outImg = opt.siftedimg_dir
# print(opt)
# matchedTemplateClass = img_match.CorrectImage(queryImagePath, templateImgDir, outImg)
# print(matchedTemplateClass)
#
# # check the pointer position and compute the num according to the degree of pointer
# if matchedTemplateClass is None:
#     raise ValueError("no find the right template class")
#
# corrected_img_path = outImg + queryImagePath.split("/")[-1]
# templateclass = find_templateclass_using_name(matchedTemplateClass)
#
# num = templateclass.degree2num(corrected_img_path)
# print(num)
