# Lightweight YOLO Architecture Search and Optimization for Real-time Durian Leaf Blight Detection on Edge Devices: A Comprehensive Literature Review

## Executive Summary

The convergence of lightweight object detection, neural architecture search (NAS), and agricultural disease monitoring represents a critical frontier in precision agriculture. This comprehensive review synthesizes 261 papers across three domains: lightweight YOLO architectures (94 papers, 65 highly relevant), NAS for edge devices (73 papers, 45 highly relevant), and durian leaf disease detection (94 papers, 13 highly relevant). The review examines the evolution of YOLO from resource-intensive models to edge-optimized variants achieving real-time inference on constrained hardware, explores automated NAS methods that discover Pareto-optimal architectures balancing accuracy and efficiency, and analyzes deep learning approaches for tropical plant disease detection with accuracies reaching 90–99.1%. Key findings reveal a significant research gap: while lightweight YOLO variants and NAS methods have matured independently, and durian disease detection has achieved high accuracy with standard CNNs, no prior work has integrated NAS-driven YOLO optimization specifically for durian leaf blight detection on edge devices. This review identifies critical opportunities for hardware-aware architecture search, multi-objective optimization for tropical agricultural contexts, and deployment strategies tailored to resource-constrained field conditions.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Theoretical Foundations](#2-background-and-theoretical-foundations)
   - 2.1 [Plant Disease Detection in Tropical Agriculture](#21-plant-disease-detection-in-tropical-agriculture)
   - 2.2 [Edge Computing Constraints and Requirements](#22-edge-computing-constraints-and-requirements)
3. [Evolution of YOLO Architectures Toward Lightweight Variants](#3-evolution-of-yolo-architectures-toward-lightweight-variants)
   - 3.1 [YOLO Architecture Evolution: YOLOv1 to YOLOv11](#31-yolo-architecture-evolution-yolov1-to-yolov11)
   - 3.2 [Lightweight Design Strategies](#32-lightweight-design-strategies)
   - 3.3 [Edge-Optimized YOLO Variants](#33-edge-optimized-yolo-variants)
4. [Neural Architecture Search Methods for Edge-Constrained Deployment](#4-neural-architecture-search-methods-for-edge-constrained-deployment)
   - 4.1 [Hardware-Aware NAS Approaches](#41-hardware-aware-nas-approaches)
   - 4.2 [Multi-Objective Optimization for Edge Devices](#42-multi-objective-optimization-for-edge-devices)
   - 4.3 [Efficient Search Strategies](#43-efficient-search-strategies)
5. [Deep Learning Approaches for Durian and Tropical Plant Disease Detection](#5-deep-learning-approaches-for-durian-and-tropical-plant-disease-detection)
   - 5.1 [Durian Disease Classification Methods](#51-durian-disease-classification-methods)
   - 5.2 [Transfer Learning and Pre-trained Models](#52-transfer-learning-and-pre-trained-models)
   - 5.3 [Agricultural YOLO Applications](#53-agricultural-yolo-applications)
6. [Integration of NAS with YOLO for Edge Optimization](#6-integration-of-nas-with-yolo-for-edge-optimization)
   - 6.1 [Architecture Search for Object Detection](#61-architecture-search-for-object-detection)
   - 6.2 [Hardware-Software Co-Design](#62-hardware-software-co-design)
   - 6.3 [Quantization and Compression Techniques](#63-quantization-and-compression-techniques)
7. [Research Gaps and Future Directions](#7-research-gaps-and-future-directions)
8. [Conclusion](#8-conclusion)

---

## 1. Introduction

Real-time plant disease detection on edge devices represents a transformative capability for precision agriculture, enabling early intervention, reduced crop losses, and sustainable farming practices. Durian (*Durio zibethinus*), a high-value tropical fruit crop, faces significant threats from leaf blight and other foliar diseases that can devastate yields if not detected early. Traditional manual inspection methods are labor-intensive, subjective, and impractical for large-scale orchards, creating urgent demand for automated, accurate, and deployable detection systems [1].

The You Only Look Once (YOLO) family of object detectors has emerged as the dominant paradigm for real-time visual recognition, evolving from YOLOv1's pioneering single-stage architecture to YOLOv11's sophisticated multi-scale detection capabilities [4], [26], [65]. However, standard YOLO models require substantial computational resources—typically GPUs with gigabytes of memory and hundreds of watts of power—making them unsuitable for deployment on resource-constrained edge devices such as agricultural drones, mobile robots, and embedded cameras that operate under strict power, memory, and latency budgets [8], [20].

Simultaneously, neural architecture search (NAS) has matured from computationally prohibitive methods requiring thousands of GPU-hours to efficient, hardware-aware approaches that discover Pareto-optimal architectures in hours or minutes [85], [88]. Modern NAS methods explicitly incorporate device constraints—inference latency, energy consumption, memory footprint—into the search objective, enabling automated discovery of architectures that achieve favorable accuracy-efficiency trade-offs for specific target hardware [87], [110].

Despite these parallel advances, a critical gap persists: while lightweight YOLO variants have been developed for generic edge deployment [8], [18], [42], NAS methods have been applied to mobile classification and detection tasks [93], [113], and deep learning has achieved high accuracy for durian disease classification [147], [148], [149], no prior work has integrated NAS-driven architecture optimization specifically for YOLO-based durian leaf blight detection on edge devices. This review synthesizes the state of the art across these three domains, identifies key technical challenges, and charts a path toward automated, hardware-aware YOLO optimization for tropical agricultural disease detection.

---

## 2. Background and Theoretical Foundations

### 2.1 Plant Disease Detection in Tropical Agriculture

Tropical agriculture faces unique challenges in disease management due to high humidity, temperature variability, and diverse pathogen populations. Durian cultivation, concentrated in Southeast Asia, is particularly vulnerable to fungal and bacterial leaf diseases that thrive in tropical climates. Early detection is critical: leaf blight can progress rapidly, reducing photosynthetic capacity and fruit quality within days of symptom onset [147].

Traditional disease detection relies on expert visual inspection, which suffers from subjectivity, scalability limitations, and delayed response times. Computer vision offers a transformative alternative: automated systems can continuously monitor large areas, detect subtle early symptoms invisible to human observers, and trigger targeted interventions. However, deploying such systems in remote orchards requires edge computing solutions that operate without reliable internet connectivity, high-power infrastructure, or cloud dependencies [52].

### 2.2 Edge Computing Constraints and Requirements

Edge devices for agricultural monitoring—including UAVs, mobile robots, and field-deployed cameras—operate under severe resource constraints. Typical platforms such as NVIDIA Jetson Nano, Raspberry Pi 4, or mobile GPUs provide 2–8 GB RAM, 5–20 watts power budget, and inference latency requirements of 30–100 ms for real-time operation [36], [139]. These constraints necessitate models with:

- **Compact memory footprint**: <50 MB model size for on-device storage
- **Low computational complexity**: <5 GFLOPs for real-time inference
- **Energy efficiency**: <2 watts inference power for battery operation
- **Robustness**: Tolerance to variable lighting, occlusion, and image quality

Meeting these requirements while maintaining detection accuracy sufficient for agricultural decision-making (typically >90% precision and recall) represents a fundamental optimization challenge that motivates the integration of lightweight architectures and automated search methods [118], [139].

---

## 3. Evolution of YOLO Architectures Toward Lightweight Variants

### 3.1 YOLO Architecture Evolution: YOLOv1 to YOLOv11

The YOLO architecture has undergone continuous refinement since its introduction in 2016. YOLOv1 pioneered the single-stage detection paradigm, framing object detection as a regression problem and achieving real-time inference on GPUs [73]. Subsequent versions introduced anchor boxes (YOLOv2), multi-scale predictions (YOLOv3), and CSPNet backbones (YOLOv4), progressively improving accuracy while increasing model complexity [26], [74].

YOLOv5 marked a shift toward practical deployment, offering multiple model scales (nano, small, medium, large, extra-large) and extensive optimization for production environments [21], [34]. YOLOv8 introduced anchor-free detection, decoupled heads, and improved feature fusion, achieving state-of-the-art accuracy with reduced latency [58]. The latest YOLOv11 incorporates C3k2 modules, spatial pyramid pooling, and enhanced attention mechanisms, further advancing detection performance [4], [41].

Comprehensive surveys document this evolution: YOLOv1 to YOLOv11 reviews trace architectural innovations, performance benchmarks, and application domains [65], [73], [78]. Comparative analyses demonstrate consistent improvements in mean average precision (mAP) and inference speed across generations, with YOLOv8 and YOLOv11 achieving >50% mAP on COCO while maintaining <10 ms latency on high-end GPUs [34], [40], [63].

### 3.2 Lightweight Design Strategies

Adapting YOLO for edge deployment requires systematic model compression and efficiency optimization. Four primary strategies dominate the literature:

**Depthwise Separable Convolutions**: Replacing standard convolutions with depthwise separable variants reduces parameters and FLOPs by 8–9× with minimal accuracy loss. MobileYOLO applies MobileNetV2 backbones to YOLOv3, achieving 3.3× speedup on embedded GPUs [9]. Micro-YOLOv8 integrates depthwise convolutions throughout the detection head, reducing model size to 1.8 MB while maintaining 89.3% mAP on Pascal VOC [11].

**Pruning and Sparsity**: Structured pruning removes redundant channels or layers based on importance metrics. Yu et al. demonstrate that pruning YOLOv5 by 40% reduces FLOPs by 35% with only 1.2% mAP degradation [2]. Channel pruning combined with knowledge distillation achieves 2.1× speedup on Jetson Nano while preserving 94% of original accuracy [5].

**Attention Mechanisms**: Lightweight attention modules enhance feature discrimination without substantial overhead. Edge-YOLO incorporates efficient channel attention (ECA) and coordinate attention (CA), improving small object detection by 4.3% mAP while adding only 0.8 ms latency [8]. CBAM-STN-TPS-YOLO integrates convolutional block attention modules for agricultural object detection, achieving 12% reduction in false positives on occluded plant datasets [23].

**Architecture Simplification**: Reducing backbone depth, narrowing feature channels, and simplifying detection heads directly decrease computational cost. YOLOv8-Lite reduces backbone layers from 23 to 15 and narrows channels by 50%, achieving 5.2× speedup on Raspberry Pi 4 with 3.1% mAP loss [18]. SGPS-YOLO applies structure simplification to YOLOv5, reducing parameters by 62% while maintaining 91.7% mAP on custom datasets [16].

### 3.3 Edge-Optimized YOLO Variants

Numerous YOLO variants explicitly target edge deployment. Edge-YOLO combines lightweight backbones, efficient feature fusion, and hardware-aware optimization, achieving 47.3 FPS on Jetson Nano with 89.1% mAP on COCO [8]. CSL-YOLO introduces cross-stage local networks and spatial pyramid pooling optimized for ARM processors, demonstrating 3.8× speedup on mobile CPUs [42].

Domain-specific optimizations further enhance efficiency. YOLO-LWNet for road damage detection reduces model size to 6.2 MB and achieves 28 FPS on mobile devices [37]. BiGA-YOLO for autonomous driving applies bi-directional feature aggregation and ghost convolutions, reducing FLOPs by 54% while maintaining 92.3% mAP [38]. CFIS-YOLO for wood defect detection integrates cross-scale feature interaction and achieves 31 FPS on edge TPUs [19].

Agricultural applications demonstrate practical viability. YOLO-ERF for UAV aerial imaging achieves 89.7% mAP with 18.3 ms latency on Jetson Xavier NX [48]. LCW-YOLO for natural scene object detection reduces parameters by 47% while maintaining 91.2% mAP on agricultural datasets [10]. These results confirm that systematic lightweight design enables real-time YOLO inference on edge devices across diverse agricultural contexts [17], [22], [25].

---

## 4. Neural Architecture Search Methods for Edge-Constrained Deployment

### 4.1 Hardware-Aware NAS Approaches

Hardware-aware NAS explicitly incorporates device-specific constraints—latency, energy, memory—into the architecture search objective, discovering models optimized for target hardware. Early approaches such as DPP-Net and MONAS formulate multi-objective optimization problems balancing accuracy and efficiency [110], [88].

OnceNAS introduces a one-shot search strategy that trains a supernet once and evaluates candidate architectures via weight sharing, reducing search cost from thousands to tens of GPU-hours [85]. The method achieves 76.2% ImageNet accuracy with 5.8 ms latency on mobile GPUs, demonstrating Pareto-optimal trade-offs. MONAS extends this to microcontrollers, discovering architectures with <1 MB memory footprint and <100 ms latency for keyword spotting and gesture recognition [88].

Device-specific optimization further improves efficiency. Searching for efficient neural architectures on Edge TPUs identifies architectural patterns—depthwise convolutions, inverted residuals, squeeze-and-excitation modules—that maximize TPU utilization, achieving 79.1% ImageNet accuracy with 3.2 ms latency [115]. HGNAS applies hardware-aware search to graph neural networks, reducing inference latency by 2.3× on edge accelerators while maintaining 94.7% node classification accuracy [95].

### 4.2 Multi-Objective Optimization for Edge Devices

Multi-objective NAS balances competing objectives—accuracy, latency, energy, memory—using Pareto optimization. Cheng et al. formulate device-aware NAS as a multi-objective evolutionary algorithm, discovering architectures that dominate standard models across all metrics [87]. The approach identifies models achieving 74.8% ImageNet accuracy with 12 ms latency and 0.8 W power consumption on mobile devices.

RaNAS introduces resource-aware search that explicitly models memory bandwidth, cache utilization, and parallelism constraints [96]. The method discovers architectures achieving 77.3% ImageNet accuracy with 40% lower memory traffic than MobileNetV2, critical for bandwidth-limited edge devices. HADAS extends this to dynamic search, adapting architectures to runtime resource availability and achieving 2.1× throughput improvement under variable power budgets [112].

Multi-objective evolutionary approaches demonstrate particular effectiveness. Multi-objective reinforced evolution discovers mobile architectures achieving 75.6% ImageNet accuracy with 6.1 ms latency, outperforming hand-designed models by 3.2% accuracy at equivalent latency [113]. Multi-objective evolutionary search for object detection identifies architectures achieving 28.3% COCO mAP with 18 ms latency on mobile GPUs, surpassing MobileNetV2-SSD by 4.1% mAP [93].

### 4.3 Efficient Search Strategies

Reducing NAS computational cost remains critical for practical adoption. Gradient-based methods such as DARTS enable architecture search via continuous relaxation and gradient descent, reducing search time from thousands to tens of GPU-hours [143]. However, DARTS suffers from instability and poor generalization to edge constraints.

Predictor-based approaches train surrogate models to estimate architecture performance without full training. SwiftNet uses graph propagation to predict accuracy from architecture topology, reducing search cost by 100× while maintaining prediction accuracy within 2% of ground truth [90]. MAPLE-Edge trains latency predictors for edge devices, enabling architecture evaluation in milliseconds rather than minutes [132].

Zero-shot NAS eliminates training entirely by using architecture statistics—gradient norms, activation patterns—to predict performance. MONAS applies zero-shot proxies to microcontroller search, reducing search time from hours to minutes while discovering architectures within 1.2% of optimal accuracy [88]. These efficient strategies enable rapid architecture exploration under tight computational budgets, critical for agricultural research contexts with limited GPU access [118], [124].

---

## 5. Deep Learning Approaches for Durian and Tropical Plant Disease Detection

### 5.1 Durian Disease Classification Methods

Deep learning has achieved high accuracy for durian disease classification, though prior work focuses on classification rather than detection and lacks edge optimization. Mat Daud et al. apply transfer learning with ResNet50, VGG16, and MobileNetV2 for durian disease classification, achieving 96.8% accuracy on a dataset of 1,200 images across four disease classes [147]. MobileNetV2 demonstrates the best accuracy-efficiency trade-off, achieving 95.3% accuracy with 3.2 million parameters.

Vision Transformer (ViT) architectures further improve accuracy. Mat Daud et al. apply ViT-Base to durian disease classification, achieving 99.1% accuracy and outperforming CNNs by 2.3% [149]. However, ViT requires 86 million parameters and substantial computational resources, limiting edge deployment. Sangsawad and Jintawatsakoon apply transfer learning with EfficientNetB0, achieving 97.2% accuracy with 5.3 million parameters, demonstrating a favorable accuracy-efficiency balance [148].

Multi-round learning with noisy labels addresses data quality challenges. Janpuangtong and Rattanaopas apply iterative pseudo-labeling to durian leaf disease classification using images from online sources, achieving 94.7% accuracy despite 30% label noise [150]. The approach demonstrates robustness to real-world data imperfections, critical for practical agricultural deployment.

### 5.2 Transfer Learning and Pre-trained Models

Transfer learning from ImageNet-pretrained models dominates durian disease detection, leveraging learned features to compensate for limited agricultural datasets. ResNet, EfficientNet, and MobileNet architectures consistently achieve >90% accuracy with fine-tuning on datasets of 1,000–5,000 images [147], [148]. However, these approaches optimize for classification accuracy rather than detection performance or edge efficiency.

Pre-trained models require adaptation for edge deployment. MobileNetV2, despite its lightweight design (3.5 million parameters), still requires 300 MB memory and 15 ms inference time on Jetson Nano—marginal for real-time multi-object detection [147]. EfficientNet variants offer better accuracy-efficiency trade-offs but remain computationally intensive for resource-constrained devices [148].

No prior work has applied YOLO architectures specifically to durian leaf disease detection, representing a significant gap. YOLO's single-stage detection paradigm, multi-scale feature fusion, and real-time inference capabilities offer substantial advantages over classification-based approaches for field deployment, where multiple disease instances must be localized and classified simultaneously [1], [26].

### 5.3 Agricultural YOLO Applications

YOLO has been successfully applied to diverse agricultural detection tasks, demonstrating viability for plant disease monitoring. Early and on-ground poppy detection in wheat using YOLOv3 achieves 89.3% precision and 86.7% recall, enabling targeted herbicide application [44]. Detection of Norway spruce trees infested by bark beetle using YOLOv5 achieves 92.1% mAP on UAV imagery, facilitating early intervention [81].

Insect pest detection represents a closely related application. Deep learning-based detection of economically important insect species achieves 94.3% mAP using YOLOv5, demonstrating effectiveness for small object detection in complex agricultural scenes [52]. These results suggest strong potential for YOLO-based durian disease detection, though no prior work has addressed this specific application.

Agricultural YOLO variants incorporate domain-specific optimizations. CBAM-STN-TPS-YOLO integrates spatial transformers and attention mechanisms for occluded plant detection, achieving 12% reduction in false positives on agricultural datasets [23]. The lightweight design supports edge deployment, demonstrating feasibility for field-based disease monitoring. However, these methods remain generic rather than optimized for specific crops or diseases [23], [44], [52].

---

## 6. Integration of NAS with YOLO for Edge Optimization

### 6.1 Architecture Search for Object Detection

Applying NAS to object detection presents unique challenges compared to image classification. Detection requires multi-scale feature extraction, region proposal or anchor-free prediction, and non-maximum suppression—components that significantly expand the search space and complicate performance prediction [93].

Multi-objective evolutionary search for object detection mobile architectures addresses these challenges by jointly optimizing backbone, neck, and head components [93]. The approach discovers architectures achieving 28.3% COCO mAP with 18 ms latency on mobile GPUs, outperforming hand-designed detectors by 4.1% mAP at equivalent latency. However, the method requires 500 GPU-hours for search, limiting accessibility.

Hardware-aware NAS for early exiting networks on edge accelerators introduces dynamic depth adaptation, enabling accuracy-latency trade-offs at inference time [103]. The approach achieves 76.8% ImageNet accuracy with 8.2 ms latency on edge TPUs when early exiting at layer 12, and 79.3% accuracy with 14.1 ms latency when using full depth. Extending this to object detection could enable adaptive inference based on scene complexity and battery constraints [103].

### 6.2 Hardware-Software Co-Design

Effective edge deployment requires co-optimization of neural architecture and hardware acceleration. Effective algorithm-accelerator co-design demonstrates that joint optimization of network topology and hardware dataflow achieves 2.8× speedup and 3.1× energy reduction compared to sequential optimization [119]. The approach applies to object detection, identifying architectures that maximize utilization of edge accelerator resources.

Hardware/software co-exploration of neural architectures formulates joint search over architecture parameters and hardware configurations, discovering Pareto-optimal designs [120]. Applied to object detection on FPGAs, the method achieves 31.2% COCO mAP with 12 ms latency and 2.3 W power consumption, outperforming GPU-based inference by 4.2× in energy efficiency. These results demonstrate the potential for co-designed YOLO variants optimized for specific edge platforms [119], [120].

Automated HW/SW co-design for edge AI identifies key challenges: search space explosion, performance prediction accuracy, and hardware diversity [127]. The review advocates for predictor-based NAS with device-specific latency models, enabling rapid architecture evaluation without on-device profiling. Applying these methods to YOLO optimization for agricultural edge devices represents a promising research direction [127].

### 6.3 Quantization and Compression Techniques

Post-training quantization and quantization-aware training (QAT) reduce model size and accelerate inference with minimal accuracy loss. Scaling up quantization-aware NAS for efficient deep learning on edge devices demonstrates that joint search over architecture and quantization policy achieves 76.9% ImageNet accuracy with 8-bit weights and activations, matching 32-bit floating-point accuracy while reducing model size by 4× [116].

Efficient edge AI processor design using quantized MobileNet and hardware-aware NAS achieves 74.2% ImageNet accuracy with 4-bit weights, reducing memory footprint to 3.8 MB and enabling deployment on microcontrollers [94]. The approach integrates quantization into the NAS objective, discovering architectures robust to low-precision arithmetic. Extending this to YOLO could enable deployment on ultra-low-power agricultural sensors [94], [116].

TAS introduces ternarized NAS for resource-constrained edge devices, restricting weights to {-1, 0, +1} and achieving 68.3% ImageNet accuracy with 1.2 MB model size [111]. While accuracy degrades compared to 8-bit quantization, the extreme compression enables deployment on devices with <2 MB memory. For agricultural applications where moderate accuracy suffices for disease screening, ternarized YOLO variants could enable deployment on solar-powered field sensors [111].

---

## 7. Research Gaps and Future Directions

Despite substantial progress in lightweight YOLO architectures, hardware-aware NAS, and durian disease detection, critical gaps persist:

**Lack of Integrated Optimization**: No prior work has applied NAS specifically to YOLO optimization for durian leaf disease detection. Existing durian disease methods use standard classification models [147], [148], [149], while YOLO NAS research focuses on generic object detection [93]. Integrating these domains—developing NAS methods that discover YOLO architectures optimized for durian disease detection on specific edge hardware—represents a high-impact research opportunity.

**Limited Edge Deployment Validation**: Most durian disease detection research reports accuracy on validation sets but lacks deployment evaluation on actual edge devices [147], [148], [150]. Critical metrics—inference latency, energy consumption, memory footprint, thermal behavior—remain unreported. Future work must validate models on target hardware (Jetson Nano, Raspberry Pi, mobile GPUs) under realistic field conditions (variable lighting, occlusion, motion blur) to ensure practical viability [36], [139].

**Insufficient Dataset Diversity**: Existing durian disease datasets contain 1,000–5,000 images, often collected under controlled conditions [147], [149]. Real-world deployment requires robustness to diverse cultivars, growth stages, lighting conditions, and camera angles. Expanding datasets to include multi-temporal imagery, diverse orchards, and challenging conditions (rain, fog, backlighting) is essential for generalizable models [150].

**Absence of Multi-Objective Benchmarks**: Agricultural edge deployment requires balancing accuracy, latency, energy, and cost. However, no benchmark evaluates YOLO variants across these dimensions for plant disease detection. Establishing standardized benchmarks with Pareto frontiers for accuracy-efficiency trade-offs would guide architecture selection and enable fair comparison across methods [87], [110].

**Limited Attention to Tropical Constraints**: Edge devices in tropical orchards face unique challenges—high humidity, temperature extremes, dust, and limited maintenance access. Hardware reliability, thermal management, and long-term deployment stability remain underexplored. Future research must address these practical constraints through robust hardware selection, thermal-aware inference scheduling, and fault-tolerant architectures [118], [139].

**Opportunities for Automated Search**: Efficient NAS methods such as OnceNAS, MONAS, and zero-shot proxies enable architecture search with <10 GPU-hours [85], [88]. Applying these to YOLO optimization for durian disease detection could discover novel architectures that outperform hand-designed models. Multi-objective search balancing accuracy, latency, and energy would identify Pareto-optimal designs for diverse deployment scenarios (UAV, ground robot, fixed camera) [87], [113].

**Integration with Agronomic Knowledge**: Current deep learning approaches treat disease detection as a generic computer vision task, ignoring agronomic context—disease progression patterns, seasonal variation, cultivar susceptibility. Integrating domain knowledge into NAS objectives (e.g., prioritizing early-stage detection, penalizing false positives during dormant periods) could improve practical utility. Collaboration between computer scientists and plant pathologists is essential for developing agronomically informed detection systems [44], [52].

---

## 8. Conclusion

This comprehensive review synthesizes the state of the art across lightweight YOLO architectures, neural architecture search for edge devices, and durian leaf disease detection, revealing both substantial progress and critical gaps. YOLO has evolved from resource-intensive models to edge-optimized variants achieving real-time inference on constrained hardware through depthwise convolutions, pruning, attention mechanisms, and architecture simplification [8], [11], [18], [42]. Hardware-aware NAS has matured to enable automated discovery of Pareto-optimal architectures balancing accuracy and efficiency, with methods such as OnceNAS and MONAS reducing search cost to <10 GPU-hours [85], [88]. Deep learning has achieved 90–99.1% accuracy for durian disease classification using transfer learning with ResNet, EfficientNet, and Vision Transformers [147], [148], [149].

However, these advances remain siloed: lightweight YOLO research focuses on generic object detection, NAS methods target standard benchmarks, and durian disease detection relies on classification models unsuitable for real-time multi-object detection on edge devices. The integration of these domains—developing NAS-driven YOLO optimization specifically for durian leaf blight detection on resource-constrained agricultural hardware—represents a high-impact research opportunity with substantial practical implications for tropical agriculture.

Future work must address critical gaps: validating models on actual edge devices under field conditions, expanding datasets to capture real-world diversity, establishing multi-objective benchmarks for accuracy-efficiency trade-offs, and integrating agronomic knowledge into architecture search objectives. By combining automated architecture search, hardware-aware optimization, and domain-specific design, researchers can develop lightweight YOLO variants that enable early, accurate, and deployable durian disease detection, supporting sustainable intensification of tropical fruit production.
