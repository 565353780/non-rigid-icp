import os
import cv2

from non_rigid_icp.Method.path import createFileFolder, removeFile

def toVideo(
    image_folder_path: str,
    save_video_file_path: str,
    fps: int=30,
    overwrite: bool=False,
) -> bool:
    if not os.path.exists(image_folder_path):
        print('[ERROR][video::toVideo]')
        print('\t image folder not exist!')
        print('\t image_folder_path:', image_folder_path)
        return False

    if os.path.exists(save_video_file_path):
        if not overwrite:
            return True

        removeFile(save_video_file_path)

    createFileFolder(save_video_file_path)

    image_file_name_list = os.listdir(image_folder_path)

    valid_image_file_name_list = []
    for image_file_name in image_file_name_list:
        if image_file_name.split('.')[-1] not in ['jpg', 'png', 'jpeg']:
          continue

        valid_image_file_name_list.append(image_file_name)

    valid_image_file_name_list = sorted(valid_image_file_name_list, key=lambda x: int(os.path.splitext(x)[0]))

    if len(valid_image_file_name_list) == 0:
        print('[ERROR][video::toVideo]')
        print('\t valid image not found!')
        print('\t image_folder_path:', image_folder_path)
        return False

    first_image_file_path = image_folder_path + valid_image_file_name_list[0]
    frame = cv2.imread(first_image_file_path)
    height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(save_video_file_path, fourcc, fps, (width, height))

    for image_file_name in valid_image_file_name_list:
        img_file_path = image_folder_path + image_file_name
        frame = cv2.imread(img_file_path)
        if frame is None:
            print('[WARN][video::toVideo]')
            print('\t can not load current frame:')
            print('\t', img_file_path)
            continue

        video_writer.write(frame)

    video_writer.release()

    return True
