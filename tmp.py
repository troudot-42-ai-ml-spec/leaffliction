import numpy as np
from plantcv import plantcv as pcv
from utils import show_grid

rand = np.random.mtrand.randint(0,100)
# rand = 26


for i in range(rand, rand + 15, 1):
    img, _, _ =  pcv.readimage(filename=f'/Users/troudot/42-cursus/leaffliction/images/Apple/Apple_rust/image ({i}).JPG', mode='rgb')
    img1, _, _ = pcv.readimage(filename=f'/Users/troudot/42-cursus/leaffliction/images/Apple/Apple_healthy/image ({i}).JPG', mode='rgb')
    img2, _, _ = pcv.readimage(filename=f'/Users/troudot/42-cursus/leaffliction/images/Apple/Apple_scab/image ({i}).JPG', mode='rgb')
    img3, _, _ = pcv.readimage(filename=f'/Users/troudot/42-cursus/leaffliction/images/Apple/Apple_Black_rot/image ({i}).JPG', mode='rgb')

    image_list = []
    labels_list = []
    for img, label in zip([img, img1, img2, img3], ['Rust', 'Healthy', 'Scab', 'Black_Rot']):
        hsv_img = pcv.rgb2gray_hsv(img, 's')
        hsv_laba = pcv.rgb2gray_lab(img, 'a')
        hsv_labb = pcv.rgb2gray_lab(img, 'b')
        hsv_labl = pcv.rgb2gray_lab(img, 'l')

        ksize =(11, 11)

        gaussian_img = pcv.gaussian_blur(img=hsv_img, ksize=ksize)
        mask_binary = pcv.threshold.binary(hsv_img, 100)
        mask_applied_binary = pcv.apply_mask(img, mask_binary, 'white')

        gaussian_img_a = pcv.gaussian_blur(img=hsv_laba, ksize=ksize)
        mask_otsu_a= pcv.threshold.otsu(gaussian_img_a, 'dark')
        mask_otsu_a = pcv.fill_holes(bin_img=mask_otsu_a)
        mask_applied_otsu_a = pcv.apply_mask(img, mask_otsu_a, 'white')

        gaussian_img_b = pcv.gaussian_blur(img=hsv_labb, ksize=ksize)
        mask_otsu_b= pcv.threshold.otsu(gaussian_img_b, 'light')
        mask_otsu_b = pcv.fill_holes(bin_img=mask_otsu_b)
        mask_applied_otsu_b = pcv.apply_mask(img, mask_otsu_b, 'white')

        # gaussian_img_l = pcv.gaussian_blur(img=hsv_labl, ksize=ksize)
        # mask_otsu_l=pcv.threshold.otsu(gaussian_img_l, 'dark')
        # mask_otsu_l = pcv.fill_holes(bin_img=mask_otsu_l)
        # mask = pcv.threshold.dual_channels(img, 'a', 'b', )
        # mask_applied_otsu_l = pcv.apply_mask(img, mask_otsu_l, 'white')


        pcv.params.sample_label = "leafA"
        analysis_image_a = pcv.analyze.size(img=img, labeled_mask=mask_otsu_a,n_labels=1)
        pcv.params.sample_label = "leafB"
        analysis_image_b = pcv.analyze.size(img=img, labeled_mask=mask_otsu_b,n_labels=1)

        image_list.append(mask_applied_otsu_a)
        image_list.append(mask_applied_otsu_b)
        # image_list.append(mask_applied_otsu_l)
        image_list.append(analysis_image_a)
        image_list.append(analysis_image_b)
        labels_list.append('otsu A ' + label)
        labels_list.append('otsu B ' + label)
        # labels_list.append('otsu L ' + label)
        labels_list.append('Ana A ' + label)
        labels_list.append('Ana B ' + label)
        obs = pcv.outputs.observations
        rotated_img = pcv.transform.rotate(img, float(obs['leafA_1']['ellipse_angle']['value']), True)
        # for trait, info in obs['leafB_1'].items():
            # print(trait, ":", info['value'], info['label'])
        # exit()
        leafa_area, leafb_area = obs["leafA_1"]["area"]['value'], obs["leafB_1"]["area"]['value']
        leafa_hull_area, leafb_hull_area = obs["leafA_1"]["convex_hull_area"]['value'], obs["leafB_1"]["convex_hull_area"]['value']
        leafa_sol_area, leafb_sol_area = obs["leafA_1"]["solidity"]['value'], obs["leafB_1"]["solidity"]['value']
        leafa_score = (leafa_hull_area - leafa_area) * leafa_sol_area
        leafb_score = (leafb_hull_area - leafb_area) * leafb_sol_area

        selected_img = mask_applied_otsu_a if leafa_sol_area > leafb_sol_area else mask_applied_otsu_b
        image_list.append(selected_img)
        labels_list.append('selected ' + label)

        # ab = pcv.logical_or(mask_otsu_b, mask_otsu_a)
        # ab_mask_applied = pcv.apply_mask(img, ab, 'black')
        # image_list.append(ab_mask_applied)
        # labels_list.append('combined ' + label)

        # sub = pcv.background_subtraction(background_image=bkgd_img, foreground_image=color_img)
        # filled_holes_mask = pcv.fill_holes(mask_otsu_a if leafa_sol_area > leafb_sol_area else mask_otsu_b)
        # filled_holes_img = pcv.apply_mask(img, filled_holes_mask, 'white')
        # image_list.append(filled_holes_img)
        # labels_list.append('filled ' + label)
        # print(obs["leafA_1"]["solidity"]['value'], obs["leafB_1"][""]['value'], )

    # if name in cfg.types:
    #     types[name].labels[root.name[separator_i+1:]] = len(files)

    # colorspace_img = pcv.visualize.colorspaces(rgb_img=img)
    # roi = pcv.roi.from_binary_image(img=img, bin_img=mask_otsu_b)
    # filtered_mask = pcv.roi.filter(mask=mask_otsu_b, roi=roi, roi_type='partial')
    # leaf, leaf_mask = pcv.object_composition(img, objects, obj_hierarchy)
    # show_grid([mask_otsu_b, analysis_image], ['normal', 'filled'])
    # show_grid(image_list, labels_list)
    # show_grid([analysis_image, hsv_laba,hsv_labb,  mask_applied_otsu_a, mask_applied_otsu_b, mask_applied_binary, img], ['size analysis','gray_lab_a','gray_lab_b', 'otsu A', 'otsu B', 'binaire', 'normal'])
    # print(f'elm n{i}')
    # pcv.plot_image(mask_applied)
