# Referensi

Semua paper yang dipakai dalam riset ini, dengan daftar penulis lengkap dan sumber aslinya. Terima kasih kepada para penulis atas karya mereka; semua hak cipta tetap milik penulis dan penerbitnya. BibTeX semua paper ada di [`referensi.bib`](referensi.bib). PDF-nya ada di [`papers/`](papers/), dikelompokkan menurut topik. Tinjauan isi tiap paper ada di [`papers/DAFTAR_PAPER.md`](papers/DAFTAR_PAPER.md), dan tabel relevansi serta best practice di [`papers/README.md`](papers/README.md).

## 0. Ordinal dan regresi semi-supervised (tahap 1–3)

| # | Paper | Penulis | Venue | Sumber | Berkas | Akses |
|---|---|---|---|---|---|---|
| 0.1 | A Semi-Supervised Ordinal Classification Method Based on Ranking Consistency Regularization | Jinxiang Chen | IEEE AINIT 2025 | [DOI](https://doi.org/10.1109/AINIT65432.2025.11035003) | `0_ordinal_regresi_ssl/Chen2025_Semi_Supervised_Ordinal_Classification_Ranking_Consistency_AINIT.pdf` | © IEEE (berbayar); PDF dari akses institusi |
| 0.2 | Ordinal Regression with Multiple Output CNN for Age Estimation | Zhenxing Niu, Mo Zhou, Le Wang, Xinbo Gao, Gang Hua | CVPR 2016 | [CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2016/html/Niu_Ordinal_Regression_With_CVPR_2016_paper.html), [DOI](https://doi.org/10.1109/CVPR.2016.532) | `0_ordinal_regresi_ssl/Niu2016_Ordinal_Regression_Multiple_Output_CNN_CVPR.pdf` | CVF Open Access (© IEEE) |
| 0.3 | Contrastive Learning for Semi-Supervised Deep Regression with Generalized Ordinal Rankings from Spectral Seriation | Ce Wang, Weihang Dai, Hanru Bai, Xiaomeng Li | arXiv preprint 2025 | [arXiv:2512.09267](https://arxiv.org/abs/2512.09267) | `0_ordinal_regresi_ssl/Wang2025_Contrastive_Learning_Spectral_Seriation_arXiv2512.09267.pdf` | terbuka (arXiv) |

## 1. Augmentasi latent untuk data tabular

| # | Paper | Penulis | Venue | Sumber | Berkas | Akses |
|---|---|---|---|---|---|---|
| 1.1 | Contrastive Mixup: Self- and Semi-Supervised Learning for Tabular Domain | Sajad Darabi, Shayan Fazeli, Ali Pazoki, Sriram Sankararaman, Majid Sarrafzadeh | arXiv preprint 2021 | [arXiv:2108.12296](https://arxiv.org/abs/2108.12296) | `1_latent_tabular/Darabi2021_Contrastive_Mixup_tabular_arXiv2108.12296.pdf` | terbuka (arXiv) |
| 1.2 | Representation Space Augmentation for Effective Self-Supervised Learning on Tabular Data | Moonjung Eo, Kyungeun Lee, Hye-Seung Cho, Dongmin Kim, Ye Seul Sim, Woohyung Lim | AAAI 2025 | [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/33265), [DOI](https://doi.org/10.1609/aaai.v39i11.33265) | `1_latent_tabular/Eo2025_RaTab_Representation_Space_Augmentation_AAAI.pdf` | © AAAI; dapat dibaca gratis di situs AAAI |
| 1.3 | Semi-Supervised Learning with Data Augmentation for Tabular Data | Junpeng Fang, Caizhi Tang, Qing Cui, Feng Zhu, Longfei Li, Jun Zhou, Wei Zhu | CIKM 2022 | [DOI](https://doi.org/10.1145/3511808.3557699) | `1_latent_tabular/Fang2022_SDAT_SSL_Data_Augmentation_Tabular_CIKM_scan.pdf` | © ACM (berbayar); salinan scan |
| 1.4 | Transfer Learning with Deep Tabular Models | Roman Levin, Valeriia Cherepanova, Avi Schwarzschild, Arpit Bansal, C. Bayan Bruss, Tom Goldstein, Andrew Gordon Wilson, Micah Goldblum | ICLR 2023 | [arXiv:2206.15306](https://arxiv.org/abs/2206.15306) | `1_latent_tabular/Levin2023_Transfer_Learning_Deep_Tabular_ICLR_arXiv2206.15306.pdf` | terbuka (arXiv) |
| 1.5 | Rethinking Data Augmentation for Tabular Data in Deep Learning | Soma Onishi, Shoya Meguro | arXiv preprint 2023 | [arXiv:2305.10308](https://arxiv.org/abs/2305.10308) | `1_latent_tabular/Onishi2023_Rethinking_Data_Augmentation_Tabular_arXiv2305.10308.pdf` | terbuka (arXiv) |
| 1.6 | SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training | Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C. Bayan Bruss, Tom Goldstein | arXiv preprint 2021 | [arXiv:2106.01342](https://arxiv.org/abs/2106.01342) | `1_latent_tabular/Somepalli2021_SAINT_arXiv2106.01342.pdf` | terbuka (arXiv) |
| 1.7 | SwitchTab: Switched Autoencoders Are Effective Tabular Learners | Jing Wu, Suiyao Chen, Qi Zhao, Renat Sergazinov, Chen Li, Shengjie Liu, Chongchao Zhao, Tianpei Xie, Hanqing Guo, Cheng Ji, Daniel Cociorva, Hakan Brunzel | AAAI 2024 | [arXiv:2401.02013](https://arxiv.org/abs/2401.02013) | `1_latent_tabular/Wu2024_SwitchTab_AAAI_arXiv2401.02013.pdf` | terbuka (arXiv) |

## 2. Augmentasi fitur/latent untuk semi-supervised (umum)

| # | Paper | Penulis | Venue | Sumber | Berkas | Akses |
|---|---|---|---|---|---|---|
| 2.1 | MODALS: Modality-agnostic Automated Data Augmentation in the Latent Space | Tsz-Him Cheung, Dit-Yan Yeung | ICLR 2021 | [ICLR 2021](https://iclr.cc/virtual/2021/poster/2589), [kode](https://github.com/jamestszhim/modals) | `2_latent_ssl_umum/Cheung2021_MODALS_Latent_Space_Augmentation_ICLR.pdf` | prosiding ICLR (terbuka) |
| 2.2 | Dataset Augmentation in Feature Space | Terrance DeVries, Graham W. Taylor | ICLR 2017 Workshop | [arXiv:1702.05538](https://arxiv.org/abs/1702.05538) | `2_latent_ssl_umum/DeVries2017_Dataset_Augmentation_in_Feature_Space_arXiv1702.05538.pdf` | terbuka (arXiv) |
| 2.3 | Revisiting Consistency Regularization for Semi-Supervised Learning | Yue Fan, Anna Kukleva, Dengxin Dai, Bernt Schiele | IJCV 2023 | [DOI](https://doi.org/10.1007/s11263-022-01723-4) | `2_latent_ssl_umum/Fan2023_Revisiting_Consistency_Regularization_FeatDistLoss_IJCV.pdf` | akses terbuka (© The Author(s)) |
| 2.4 | FeatMatch: Feature-Based Augmentation for Semi-Supervised Learning | Chia-Wen Kuo, Chih-Yao Ma, Jia-Bin Huang, Zsolt Kira | ECCV 2020 | [arXiv:2007.08505](https://arxiv.org/abs/2007.08505) | `2_latent_ssl_umum/Kuo2020_FeatMatch_ECCV_arXiv2007.08505.pdf` | terbuka (arXiv) |
| 2.5 | Tied-Augment: Controlling Representation Similarity Improves Data Augmentation | Emirhan Kurtulus, Zichao Li, Yann Dauphin, Ekin Dogus Cubuk | arXiv preprint 2023 | [arXiv:2305.13520](https://arxiv.org/abs/2305.13520) | `2_latent_ssl_umum/Kurtulus2023_Tied-Augment_Feature_Similarity_arXiv2305.13520.pdf` | terbuka (arXiv) |
| 2.6 | Semi-supervised Regression by Preserving Ranking Relationships between Close Unlabeled Samples | Ximing Li, Jiaxuan Jiang, Changchun Li, You Lu, Renchu Guan | AAAI 2026 | [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/39487), [DOI](https://doi.org/10.1609/aaai.v40i28.39487), [kode](https://github.com/changchunli/SSR-RCUS) | `2_latent_ssl_umum/Li2026_SSR-RCUS_Ranking_Close_Unlabeled_Samples_AAAI.pdf` | © AAAI; dapat dibaca gratis di situs AAAI |
| 2.7 | Regularization with Latent Space Virtual Adversarial Training | Genki Osada, Budrul Ahsan, Revoti Prasad Bora, Takashi Nishide | ECCV 2020 | [arXiv:2011.13181](https://arxiv.org/abs/2011.13181) | `2_latent_ssl_umum/Osada2020_Latent_Space_Virtual_Adversarial_Training_arXiv2011.13181.pdf` | terbuka (arXiv) |
| 2.8 | Semi-Supervised Semantic Segmentation with Cross-Consistency Training | Yassine Ouali, Céline Hudelot, Myriam Tami | CVPR 2020 | [arXiv:2003.09005](https://arxiv.org/abs/2003.09005) | `2_latent_ssl_umum/Ouali2020_Cross_Consistency_Training_CVPR_arXiv2003.09005.pdf` | terbuka (arXiv) |
| 2.9 | Manifold Mixup: Better Representations by Interpolating Hidden States | Vikas Verma, Alex Lamb, Christopher Beckham, Amir Najafi, Ioannis Mitliagkas, Aaron Courville, David Lopez-Paz, Yoshua Bengio | ICML 2019 | [arXiv:1806.05236](https://arxiv.org/abs/1806.05236) | `2_latent_ssl_umum/Verma2019_Manifold_Mixup_ICML_arXiv1806.05236.pdf` | terbuka (arXiv) |
| 2.10 | Implicit Semantic Data Augmentation for Deep Networks | Yulin Wang, Xuran Pan, Shiji Song, Hong Zhang, Cheng Wu, Gao Huang | NeurIPS 2019 | [arXiv:1909.12220](https://arxiv.org/abs/1909.12220) | `2_latent_ssl_umum/Wang2019_ISDA_Implicit_Semantic_Data_Augmentation_NeurIPS_arXiv1909.12220.pdf` | terbuka (arXiv) |
| 2.11 | Unified Complementary Learning with Feature Perturbation for Semi-Supervised Learning | Ke Wang, Jie Yang, Yunfei Guo, Anke Xue | Algorithms 2026 | [DOI](https://doi.org/10.3390/a19010056) | `2_latent_ssl_umum/Wang2026_CLFP_Complementary_Learning_Feature_Perturbation_Algorithms.pdf` | akses terbuka (CC BY, MDPI) |
| 2.12 | Image-Feature Weak-to-Strong Consistency: An Enhanced Paradigm for Semi-Supervised Learning | Zhiyu Wu, Jinshi Cui | ECCV 2024 | [arXiv:2408.12614](https://arxiv.org/abs/2408.12614) | `2_latent_ssl_umum/Wu2024_Image_Feature_Weak_to_Strong_Consistency_ECCV_arXiv2408.12614.pdf` | terbuka (arXiv) |
| 2.13 | Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation | Lihe Yang, Lei Qi, Litong Feng, Wayne Zhang, Yinghuan Shi | CVPR 2023 | [arXiv:2208.09910](https://arxiv.org/abs/2208.09910) | `2_latent_ssl_umum/Yang2023_UniMatch_feature_perturbation_CVPR_arXiv2208.09910.pdf` | terbuka (arXiv) |
| 2.14 | UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation | Lihe Yang, Zhen Zhao, Hengshuang Zhao | arXiv preprint 2024 | [arXiv:2410.10777](https://arxiv.org/abs/2410.10777) | `2_latent_ssl_umum/Yang2025_UniMatch_V2_arXiv2410.10777.pdf` | terbuka (arXiv) |

## 3. Augmentasi dan SSL tabular

| # | Paper | Penulis | Venue | Sumber | Berkas | Akses |
|---|---|---|---|---|---|---|
| 3.1 | SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption | Dara Bahri, Heinrich Jiang, Yi Tay, Donald Metzler | ICLR 2022 | [arXiv:2106.15147](https://arxiv.org/abs/2106.15147) | `3_tabular_ssl_augmentasi/Bahri2022_SCARF_ICLR_arXiv2106.15147.pdf` | terbuka (arXiv) |
| 3.2 | Tabular Data Contrastive Learning via Class-Conditioned and Feature-Correlation Based Augmentation | Wei Cui, Rasa Hosseinzadeh, Junwei Ma, Tongzi Wu, Yi Sui, Keyvan Golestan | arXiv preprint 2024 | [arXiv:2404.17489](https://arxiv.org/abs/2404.17489) | `3_tabular_ssl_augmentasi/Cui2024_Class_Conditioned_Feature_Correlation_Augmentation_arXiv2404.17489.pdf` | terbuka (arXiv) |
| 3.3 | Mutual Information-Guided Corruption for Improved Self-Supervised Representation Learning in Tabular Data | Michael Lawson, Emerald Sy, Zhang Kehui, Raymond H. F. Chan, Kannie W. Y. Chan, Rosa H. M. Chan | ICML 2026 Workshop on Foundation Models for Structured Data | [ICML 2026](https://icml.cc/virtual/2026/71518) | `3_tabular_ssl_augmentasi/Lawson2026_MI_Guided_Corruption_Tabular_SSL.pdf` | © penulis (workshop ICML 2026) |
| 3.4 | Deep Learning within Tabular Data: Foundations, Challenges, Advances and Future Directions | Weijieying Ren, Tianxiang Zhao, Yuqing Huang, Vasant Honavar | arXiv preprint 2025 | [arXiv:2501.03540](https://arxiv.org/abs/2501.03540) | `3_tabular_ssl_augmentasi/Ren2025_Deep_Learning_Tabular_Survey_arXiv2501.03540.pdf` | terbuka (arXiv) |
| 3.5 | T-JEPA: Augmentation-Free Self-Supervised Learning for Tabular Data | Hugo Thimonier, José Lucas De Melo Costa, Fabrice Popineau, Arpad Rimmel, Bich-Liên Doan | ICLR 2025 | [arXiv:2410.05016](https://arxiv.org/abs/2410.05016) | `3_tabular_ssl_augmentasi/Thimonier2024_T-JEPA_augmentation_free_arXiv2410.05016.pdf` | terbuka (arXiv) |
| 3.6 | A Survey on Self-Supervised Learning for Non-Sequential Tabular Data | Wei-Yao Wang, Wei-Wei Du, Derek Xu, Wei Wang, Wen-Chih Peng | ACML 2024 | [arXiv:2402.01204](https://arxiv.org/abs/2402.01204) | `3_tabular_ssl_augmentasi/Wang2024_Survey_SSL_NonSequential_Tabular_arXiv2402.01204.pdf` | terbuka (arXiv) |

## 4. Backbone, adapter, dan metode dasar

| # | Paper | Penulis | Venue | Sumber | Berkas | Akses |
|---|---|---|---|---|---|---|
| 4.1 | AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition | Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, Ping Luo | NeurIPS 2022 | [arXiv:2205.13535](https://arxiv.org/abs/2205.13535) | `4_backbone_adapter/Chen2022_AdaptFormer_NeurIPS_arXiv2205.13535.pdf` | terbuka (arXiv) |
| 4.2 | TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling | Yury Gorishniy, Akim Kotelnikov, Artem Babenko | ICLR 2025 | [arXiv:2410.24210](https://arxiv.org/abs/2410.24210) | `4_backbone_adapter/Gorishniy2025_TabM_ICLR_arXiv2410.24210.pdf` | terbuka (arXiv) |
| 4.3 | RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier | Pin-Yen Huang, Szu-Wei Fu, Yu Tsao | NeurIPS 2024 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/c26a8494fe31695db965ae8b7244b7c1-Paper-Conference.pdf), [arXiv:2410.22124](https://arxiv.org/abs/2410.22124), [kode](https://github.com/pm25/Semi-Supervised-Regression), BibTeX: `4_backbone_adapter/Huang2024_RankUp_NeurIPS.bib` | `4_backbone_adapter/Huang2024_RankUp_NeurIPS.pdf` | prosiding NeurIPS (terbuka) |
| 4.4 | XTab: Cross-table Pretraining for Tabular Transformers | Bingzhao Zhu, Xingjian Shi, Nick Erickson, Mu Li, George Karypis, Mahsa Shoaran | ICML 2023 | [arXiv:2305.06090](https://arxiv.org/abs/2305.06090) | `4_backbone_adapter/Zhu2023_XTab_pretrained_tabular_ICML_arXiv2305.06090.pdf` | terbuka (arXiv) |

## 5. Prior art augmentasi sudut / cosine

| # | Paper | Penulis | Venue | Sumber | Berkas | Akses |
|---|---|---|---|---|---|---|
| 5.1 | Rotation Regularization Without Rotation | Takumi Kobayashi | ECCV 2022 | [ECVA](https://ecva.net/papers/eccv_2022/papers_ECCV/html/7735_ECCV_2022_paper.php), [arXiv:2207.10023](https://arxiv.org/abs/2207.10023), [DOI](https://doi.org/10.1007/978-3-031-19806-9_37), [kode](https://github.com/tk1980/StatRot) | `5_angular_prior_art/Kobayashi2022_Rotation_Regularization_Without_Rotation_ECCV.pdf` | tersedia di situs ECVA (© Springer) |

## Kode RankUp resmi

`Semi-Supervised-Regression/` adalah salinan kode resmi RankUp, berlisensi MIT (lisensinya ada di dalam folder).

| | |
| Repo asal | https://github.com/pm25/Semi-Supervised-Regression |
| Commit | `7ed6d267783d9dc8f99ae8188b126d125bf3ab79` |
| Perubahan | Tidak ada. Riwayat git-nya tidak disertakan |

## Catatan hak cipta

Ada 35 PDF di repo ini. Kolom *Akses* mencatat status tiap paper menurut keterangan di halaman pertama PDF-nya atau di situs sumbernya; ini bukan nasihat hukum. Preprint arXiv dan prosiding terbuka lazim dibagikan ulang. Dua paper berikut berhak cipta penerbit dan berbayar, jadi risikonya paling besar kalau repo ini publik:

- `referensi/papers/0_ordinal_regresi_ssl/Chen2025_Semi_Supervised_Ordinal_Classification_Ranking_Consistency_AINIT.pdf` (IEEE; PDF-nya memuat keterangan akses institusi)
- `referensi/papers/1_latent_tabular/Fang2022_SDAT_SSL_Data_Augmentation_Tabular_CIKM_scan.pdf` (ACM)

Dua paper AAAI (RaTab dan SSR-RCUS) juga berhak cipta AAAI, tetapi dapat dibaca gratis di situs AAAI. Untuk mengecualikan paper tertentu dari git, tambahkan jalurnya ke `.gitignore` di root repo.
