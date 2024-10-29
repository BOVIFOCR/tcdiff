import os, sys
import glob
import shutil
import numpy as np
from abc import ABCMeta, abstractmethod
import cv2


class DatasetOrganizer:
    def __init__(self, path_parent_dir='/datasets1/bjgbiesseck', output_dataset_name='MICA'):
        self.dataset_name = ''
        self.path_parent_dir = path_parent_dir
        self.subpath_input_dir = ''
        self.input_file_ext = ['']
        self.subpath_flame_parameters = ''
        self.subpath_registrations = ''
        self.subpath_output_images = ''
        self.output_dataset_name = output_dataset_name


    def load_image_paths_from_npy(self, npy_path=''):
        data = np.load(npy_path, allow_pickle=True).item()
        return data

    def concat_paths(self):
        self.subpath_npy_paths_mica = '../image_paths/' + self.dataset_name + '.npy'
        self.subpath_flame_parameters = self.output_dataset_name + '/' + self.dataset_name + '/FLAME_parameters'
        self.subpath_registrations = self.output_dataset_name + '/' + self.dataset_name + '/registrations'
        self.subpath_output_images = self.output_dataset_name + '/' + self.dataset_name + '/images'


        self.path_original_images  = self.path_parent_dir + '/' + self.subpath_input_dir
        self.path_flame_parameters = self.path_parent_dir + '/' + self.subpath_flame_parameters
        self.path_registrations    = self.path_parent_dir + '/' + self.subpath_registrations
        self.path_output_images    = self.path_parent_dir + '/' + self.subpath_output_images
        self.path_npy_paths_mica   = os.path.dirname(os.path.realpath(__file__)) + '/' + self.subpath_npy_paths_mica

    def print_folder_paths(self):
        print('self.path_original_images:', self.path_original_images)
        print('self.path_flame_parameters:', self.path_flame_parameters)
        print('self.path_registrations:', self.path_registrations)
        print('self.path_output_images:', self.path_output_images)
        print('self.path_npy_paths_mica:', self.path_npy_paths_mica)

    def create_subjects_folders(self, parent_dir='', subfolder_names=[]):
        for subfolder in subfolder_names:
            path_folder = parent_dir + '/' + subfolder

            os.makedirs(path_folder, exist_ok=True)

    def is_found_file_valid(self, subj, path_file):
        if '/'+subj in path_file or subj+'/' in path_file:
            return True
        return False

    def organize(self):
        paths_dict = self.load_image_paths_from_npy(self.path_npy_paths_mica)
        subj_names = list(paths_dict.keys())



        for i, subj in enumerate(subj_names):
            output_imgs = paths_dict[subj][0]    # ['M1044/M1044_001.jpg', 'M1044/M1044_002.jpg', etc]
            input_npz =   paths_dict[subj][1]    # 'M1044/m1044_NH.npz'





            for j, out_img in enumerate(output_imgs):
                file_name_to_search = out_img.split('/')[-1].split('.')[0]
                print('subj:', subj, '('+str(i+1)+'/'+str(len(subj_names))+')    out_img:', out_img, '    file_name_to_search:', file_name_to_search, '('+str(j+1)+'/'+str(len(output_imgs))+')')
                pattern_file_to_search = self.path_original_images + '/**/' + file_name_to_search + '.*'
                print('    pattern_file_to_search:', pattern_file_to_search)
                found_file = glob.glob(pattern_file_to_search, recursive=True)
                found_file = [f for f in found_file if self.is_found_file_valid(subj, f)]                  # Check if path contains subject name


                found_file = [f for ext in self.input_file_ext for f in found_file if f.endswith(ext)]    # Check extension of found files

                if len(found_file) == 0:
                    print('Error, file not found:', file_name_to_search)
                    sys.exit(0)
                elif len(found_file) > 1:
                    print('Error, multiple files found:', found_file)
                    sys.exit(0)
                found_file = found_file[0]
                print('    found_file:', '\''+found_file+'\'')



                if found_file.endswith(out_img.split('.')[-1]):
                    output_file = self.path_output_images + '/' + subj + '/' + out_img.split('/')[-1]
                else:
                    output_file = self.path_output_images + '/' + subj + '/' + file_name_to_search + '.jpg'

                assert output_file != found_file
                os.makedirs('/'.join(output_file.split('/')[:-1]), exist_ok=True)
                print('    copying output_file:', output_file)

                if found_file.endswith('.jpg') or found_file.endswith('.JPG') or \
                                                  found_file.endswith('.jpeg') or found_file.endswith('.JPEG') or \
                                                  found_file.endswith('.png') or found_file.endswith('.PNG'):
                    shutil.copyfile(found_file, output_file)
                else:
                    img_data = cv2.imread(found_file)
                    cv2.imwrite(output_file, img_data)

                if not os.path.exists(output_file):
                    print('Error, file not copied:', output_file)
                    sys.exit(0)



                print('-----------------')





class DatasetOrganizer_FRGCv2(DatasetOrganizer):
    def __init__(self, path_parent_dir='/datasets1/bjgbiesseck', output_dataset_name='MICA'):

        super().__init__(path_parent_dir, output_dataset_name)
        self.dataset_name = 'FRGC_v2'
        self.subpath_input_dir = 'FRGCv2.0/FRGC-2.0-dist/nd1'
        self.input_file_ext = ['.jpg', 'JPG', '.ppm']

        self.concat_paths()



class DatasetOrganizer_Stirling(DatasetOrganizer):
    def __init__(self, path_parent_dir='/datasets1/bjgbiesseck', output_dataset_name='MICA'):

        super().__init__(path_parent_dir, output_dataset_name)
        self.dataset_name = 'STIRLING'
        self.subpath_input_dir = 'Stirling-ESRC_2D/Subset_2D_FG2018/HQ'
        self.input_file_ext = ['.jpg']

        self.concat_paths()



class DatasetOrganizer_FaceWarehouse(DatasetOrganizer):
    def __init__(self, path_parent_dir='/datasets1/bjgbiesseck', output_dataset_name='MICA'):

        super().__init__(path_parent_dir, output_dataset_name)
        self.dataset_name = 'FACEWAREHOUSE'
        self.subpath_input_dir = 'FaceWarehouse/FaceWarehouse_Data_0.part1'
        self.input_file_ext = ['.png']

        self.concat_paths()



class DatasetOrganizer_LYHM(DatasetOrganizer):
    def __init__(self, path_parent_dir='/datasets1/bjgbiesseck', output_dataset_name='MICA'):

        super().__init__(path_parent_dir, output_dataset_name)
        self.dataset_name = 'LYHM'
        self.subpath_input_dir = 'headspacePngTka/subjects'
        self.input_file_ext = ['.png']

        self.concat_paths()


    def is_found_file_valid(self, subj, path_file):
        subj_from_path = path_file.split('/')[-3]
        if ('/'+subj in path_file or subj+'/' in path_file) and subj == subj_from_path:
            return True
        return False



class DatasetOrganizer_Florence(DatasetOrganizer):
    def __init__(self, path_parent_dir='/datasets1/bjgbiesseck', output_dataset_name='MICA'):

        super().__init__(path_parent_dir, output_dataset_name)
        self.dataset_name = 'FLORENCE'
        self.subpath_input_dir = 'FlorenceFace/Original'

        self.input_file_ext = ['Indoor-Cooperative.mjpg', 'Indoor-Cooperative.mjpeg', 'Indoor-Cooperative.mpg']

        self.concat_paths()
    
    def count_total_frames_manual(self, path_video=''):
        video = cv2.VideoCapture(path_video)
        count = 0
        while True:
            (hasNext, frame) = video.read()
            if not hasNext:
                break

            count += 1

        return count


    def organize(self):
        import cv2
        paths_dict = self.load_image_paths_from_npy(self.path_npy_paths_mica)
        subj_names = sorted(list(paths_dict.keys()))



        for i, subj in enumerate(subj_names):

            output_imgs = sorted(paths_dict[subj][0])    # ['subject_53/frame_1141.jpg', 'subject_53/frame_0967.jpg', etc]
            input_npz =   paths_dict[subj][1]            # ''subject_53/110616114712.npz'



            
            for input_file in self.input_file_ext:
                file_name_to_search = input_file


                pattern_file_to_search = self.path_original_images + '/' + subj + '/**/*' + file_name_to_search
                print('subj:', subj, '('+str(i+1)+'/'+str(len(subj_names))+'    file_name_to_search:', file_name_to_search)
                print('    pattern_file_to_search:', pattern_file_to_search)
                
                found_file = glob.glob(pattern_file_to_search, recursive=True)
                found_file = [f for f in found_file if self.is_found_file_valid(subj, f)]                  # Check if path contains subject name
                found_file = [f for ext in self.input_file_ext for f in found_file if f.endswith(ext)]     # Check extension of found files

                if len(found_file) > 0:
                    break
            
            if len(found_file) == 0:
                print('Error, file not found:', file_name_to_search)
                sys.exit(0)
            elif len(found_file) > 1:
                print('Error, multiple files found:', found_file)
                sys.exit(0)
            found_file = found_file[0]




            num_total_frames_video = self.count_total_frames_manual(found_file)
            video_reader = cv2.VideoCapture(found_file)
            count_frame = 0


            for j, out_img in enumerate(output_imgs):
                output_file = self.path_output_images + '/' + out_img
                target_frame_num = int(out_img.split('/')[-1].split('_')[-1].split('.')[0])
                print('subj:', subj, '('+str(i+1)+'/'+str(len(subj_names))+')    out_img:', out_img, '    file_name_to_search:', file_name_to_search, '('+str(j+1)+'/'+str(len(output_imgs))+')')
                print('    pattern_file_to_search:', pattern_file_to_search)
                print('    found_file:', '\''+found_file+'\'')
                print('    target_frame_num:', target_frame_num, '    num_total_frames_video:', num_total_frames_video)
                print('    output_file:', output_file)
                assert output_file != found_file

                print('    Searching frame...')
                while True:
                    (hasNext, frame) = video_reader.read()
                                            
                    count_frame += 1


                    if count_frame == target_frame_num:
                        os.makedirs('/'.join(output_file.split('/')[:-1]), exist_ok=True)
                        print('    saving frame:', output_file)
                        cv2.imwrite(output_file, frame)
                        break
                    
                    if not hasNext:
                        break
                    

                if not os.path.exists(output_file):
                    print('Error, file not copied:', output_file)
                    sys.exit(0)



                print('-----------------')




np.random.seed(42)

if __name__ == '__main__':

    path_parent_dir = '/datasets1/bjgbiesseck'
    output_dataset_name = 'MICA'





    dataset_org = DatasetOrganizer_Florence(path_parent_dir, output_dataset_name)


    dataset_org.organize()
