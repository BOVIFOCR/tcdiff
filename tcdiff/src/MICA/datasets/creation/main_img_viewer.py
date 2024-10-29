


import os, sys
import cv2
import numpy as np

import glob
import matplotlib.pyplot as plt





if __name__ == '__main__':







    folders = [
        




















































































































































































































































































































































































































































































































































        
























































































































































        
        

        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1001',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1002',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1003',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1004',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1005',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1006',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1007',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1008',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1009',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1010',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1011',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1012',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1013',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1014',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1015',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1016',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1017',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1019',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1020',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1021',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1022',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1023',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1024',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1025',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1026',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1027',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1028',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1029',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1030',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1031',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1032',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1034',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1035',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1036',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1037',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1038',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1039',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1040',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1041',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1042',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1043',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1045',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1046',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1047',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1048',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1049',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1050',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1051',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1052',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1053',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1054',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1000',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1001',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1002',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1003',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1004',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1005',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1006',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1007',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1008',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1009',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1010',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1011',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1012',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1013',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1014',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1015',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1016',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1017',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1018',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1019',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1020',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1021',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1022',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1023',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1024',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1025',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1026',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1027',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1028',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1029',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1030',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1031',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1032',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1033',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1034',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1035',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1036',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1037',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1038',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1039',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1040',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1041',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1042',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1043',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1044',
        




    ]

    img_exts = ['.jpg', '.png']
    arc_img_exts = ['.npy']

    all_img_paths = []
    all_arcface_img_paths = []

    num_rows = len(folders)
    num_cols = 0
    img_size = (0, 0)
    dist_between_rows = 20

    print('Searching files...')
    for i, folder in enumerate(folders):
        print(f'Searching files - folder {i}/{len(folders)-1}: {folder}')
        img_paths = [ glob.glob(folder + '/*' + img_ext) for img_ext in img_exts ]
        arcface_img_paths = [ glob.glob(folder + '/*' + arc_img_ext) for arc_img_ext in arc_img_exts  ]


        
        img_paths = sorted([ p for path_list in img_paths for p in path_list ])
        arcface_img_paths = sorted([ p for path_list in arcface_img_paths for p in path_list ])
        assert len(img_paths) ==  len(arcface_img_paths)

        if len(img_paths) > num_cols:
            num_cols = len(img_paths) + 1



        imsize = np.load(arcface_img_paths[0]).shape
        if imsize[1] > img_size[0] and imsize[2] > img_size[1]:
            img_size = imsize

        all_img_paths.append(img_paths)
        all_arcface_img_paths.append(arcface_img_paths)
        





    print('num_rows:', num_rows)
    print('num_cols:', num_cols)
    print('img_size:', img_size)






    scale_factor = 2.0
    height = scale_factor * num_rows
    width = scale_factor * num_cols
    plt.rcParams["figure.figsize"] = [width, height]
    plt.rcParams["figure.autolayout"] = True
    print('Making initial figure...')
    fig = plt.figure(constrained_layout=False)
    ax_array = fig.subplots(num_rows, num_cols, squeeze=False)
    for row in range(num_rows):
        for col in range(num_cols):
            ax_array[row, col].axis('off')


            
    print('-----------------------')
    for row in range(num_rows):
        folder, arcface_img_paths = folders[row], all_arcface_img_paths[row]
        dataset_name = folder.split('/')[-3]
        subj_name = folder.split('/')[-1]



        ax_array[row, 0].text(0.5, 0.5, dataset_name+'\n'+subj_name, fontsize=7)

        for col in range(1, len(arcface_img_paths)):
            print('Adding image to figure...')
            print(f'row: {row}/{num_rows-1} - col: {col}/{len(arcface_img_paths)-1}')
            arcface_image_path = arcface_img_paths[col]

            file_name = arcface_image_path.split('/')[-1]   

            print(f'arcface_image_path: {arcface_image_path}')
            arcface_image = np.load(arcface_image_path)
            arcface_image -= arcface_image.min()
            arcface_image /= 2.0
            arcface_image = arcface_image.transpose(1, 2, 0)
            print(f'arcface_image.shape: {arcface_image.shape}')















            ax_array[row, col].imshow(arcface_image)


            ax_array[row, col].set_title(file_name, size=7)
            
            print('-----------------------')

    plt.subplots_adjust(left=0.1, right=0.9,
                        bottom=0.1, top=0.9,                        
                        wspace=0.0,
                        hspace=0.6)
    
    file_path = '/home/biesseck/GitHub/ImgViewer_Python_OpenCV/view.png'
    print('Saving figure:', file_path, '...')
    plt.savefig(file_path, bbox_inches="tight", pad_inches = 0)


    print('Finished')




'''
if __name__ == '__main__':
    





















    folder = '/datasets1/bjgbiesseck/MICA/LYHM/_arcface_input/00013'
    img_ext = '.png'
    arc_img_ext = '.npy'

    img_paths = glob.glob(folder + '/*' + img_ext)


    for image_path in img_paths:



        arcface_image_path = image_path.replace(img_ext, arc_img_ext)

        image = cv2.imread(image_path)
        image = image.astype(np.float32) / 255.

        arcface_image = np.load(arcface_image_path)

        arcface_image = arcface_image.transpose(1, 2, 0)
        arcface_image = cv2.cvtColor(arcface_image, cv2.COLOR_RGB2BGR)
        
        image_hist, image_bin_edges     = np.histogram(image, bins=20, range=(-1.0,1.0))
        arcface_hist, arcface_bin_edges = np.histogram(arcface_image, bins=20, range=(-1.0,1.0))
        



        cv2.imshow('image', image)
        cv2.imshow('arcface_image', arcface_image)
        print('image.shape:', image.shape)
        print('arcface_image.shape:', arcface_image.shape)
        print('image.min():', image.min(), '    image.max():', image.max())
        print('arcface_image.min():', arcface_image.min(), '    arcface_image.max():', arcface_image.max())
        print('arcface_hist:', arcface_hist)
        
        print('---------')

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
'''
            



'''
if __name__ == '__main__':
    



    image_path =         '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/02463/02463d170.jpg'
    arcface_image_path = '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/02463/02463d170.npy'

    image = cv2.imread(image_path)
    image = image.astype(np.float32) / 255.

    arcface_image = np.load(arcface_image_path)
    arcface_image -= arcface_image.min()
    arcface_image = arcface_image.transpose(1, 2, 0)
    
    
    cv2.imshow('image', image)
    cv2.imshow('arcface_image', arcface_image)
    print('image.shape:', image.shape)
    print('arcface_image:', arcface_image)
    print('arcface_image.shape:', arcface_image.shape)
    cv2.waitKey(0)
'''
