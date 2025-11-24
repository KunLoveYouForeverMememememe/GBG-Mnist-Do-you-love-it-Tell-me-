# GBG-Mnist-Do-you-love-it-Tell-me-
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
# 引入 sklearn 自带的小型手写数字数据集，运行速度极快
from sklearn.datasets import load_digits

# --- 全局设置 ---
# 开启 PLOT_ITERATION，你可以看到粒球是如何一步步划分的！
PLOT_ITERATION = True


# ----------------


# ----------------- 核心算法函数（微调） -----------------

def get_label_and_purity(gb):
    """计算粒球的标签和纯度"""
    labels = gb[:, 0]
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1 or gb.shape[0] == 0:
        return labels[0] if labels.size > 0 else -1, 1.0

    label_counts = {label: np.sum(labels == label) for label in unique_labels}
    max_count = max(label_counts.values())
    max_label = [label for label, count in label_counts.items() if count == max_count][0]
    purity = max_count / gb.shape[0]

    return max_label, purity


def calculate_center_and_radius(gb):
    """计算粒球中心和半径"""
    data_no_label = gb[:, 1:]

    if data_no_label.shape[0] == 0:
        # 如果粒球为空，返回零中心和零半径
        return np.zeros(data_no_label.shape[1]), 0.0

    center = data_no_label.mean(axis=0)
    distances = np.sqrt(np.sum((data_no_label - center) ** 2, axis=1))
    radius_mean = np.mean(distances)

    return center, radius_mean


def calculate_distances(data, p):
    """计算距离"""
    return np.sqrt(np.sum((data - p) ** 2))


def plot_gb(granular_ball_list, title="Granular Ball Visualization", show_plot=True):
    """绘制粒球、中心和数据点"""
    color_map = {0: 'red', 1: 'black', 2: 'blue', 3: 'green', 4: 'gold',
                 5: 'cyan', 6: 'magenta', 7: 'peru', 8: 'pink', 9: 'orange', -1: 'gray'}

    # 使用当前活动的 Figure (用于迭代绘图) 或新建 Figure (用于最终绘图)
    if show_plot:
        plt.figure(figsize=(10, 8))

    plt.clf()  # 清除当前图表内容，确保迭代图的清晰

    all_data = [gb for gb in granular_ball_list if gb.shape[0] > 0]

    # 1. 绘制数据点 (散点图)
    if all_data:
        combined_data = np.concatenate(all_data, axis=0)
        x_coords = combined_data[:, 1]
        y_coords = combined_data[:, 2]
        labels = combined_data[:, 0].astype(int)

        for label_val in np.unique(labels):
            idx = (labels == label_val)
            plt.plot(x_coords[idx], y_coords[idx], '.', color=color_map.get(label_val, 'gray'),
                     markersize=5, alpha=0.5, label=f'Class {label_val}')

    # 2. 绘制粒球边界和中心
    for granular_ball in granular_ball_list:
        if granular_ball.shape[0] == 0: continue

        label, purity = get_label_and_purity(granular_ball)
        center, radius = calculate_center_and_radius(granular_ball)

        center_2d = center[:2]

        # 绘制圆圈 (粒球边界)
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = center_2d[0] + radius * np.cos(theta)
        y = center_2d[1] + radius * np.sin(theta)
        plt.plot(x, y, color_map.get(int(label), 'gray'), linewidth=1.5, alpha=0.8, linestyle='--')

        # 绘制中心点
        plt.plot(center_2d[0], center_2d[1], 'x', color='w', markersize=8, markeredgewidth=3, zorder=5)
        plt.plot(center_2d[0], center_2d[1], 'x', color=color_map.get(int(label), 'gray'), markersize=6,
                 markeredgewidth=2, zorder=6)

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(markerscale=1.5, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    if show_plot:
        plt.show()
    else:
        plt.draw()
        plt.pause(0.05)  # 极短暂停，用于动态更新图表


# 粒球分裂条件判定 (优化迭代绘图逻辑)
def splits(purity_init, gb_dict, iteration=0):
    gb_dict_new = {}
    gb_dict_temp = {}
    first = True
    purity_init_temp = 0

    while True:
        if len(gb_dict_new) == 0:
            gb_dict_temp = gb_dict.copy()
        else:
            gb_dict_temp = gb_dict_new.copy()
            gb_dict_new = {}

        ball_number_1 = len(gb_dict_temp)

        for key in gb_dict_temp.keys():
            gb_single = {key: gb_dict_temp[key]}
            gb = gb_single[key][0]
            p = get_label_and_purity(gb)[1]

            if len(gb) <= 1:
                gb_dict_new.update(gb_single)
                continue

            gb_single_temp = gb_single.copy()
            gb_dict_re = splits_ball(gb_single).copy()

            if first:
                for key0 in gb_dict_re.keys():
                    purity_init_temp = max(purity_init_temp, get_label_and_purity(gb_dict_re[key0][0])[1])
                purity_init = purity_init_temp
                first = False

            weight_p = 0
            for key0 in gb_dict_re.keys():
                weight_p += get_label_and_purity(gb_dict_re[key0][0])[1] * (len(gb_dict_re[key0][0]) / len(gb))

            if p <= purity_init or weight_p > p:
                gb_dict_new.update(gb_dict_re)
            else:
                gb_dict_new.update(gb_single_temp)

        gb_dict_new = isOverlap(gb_dict_new)
        ball_number_2 = len(gb_dict_new)

        # 🌟 迭代可视化：如果粒球数量发生变化，立刻绘制
        if PLOT_ITERATION and ball_number_1 != ball_number_2:
            iteration += 1
            temp_ball_list = [gb_dict_new[key][0] for key in gb_dict_new.keys()]
            plot_gb(temp_ball_list, title=f"Splitting Iteration {iteration}: {ball_number_2} Granular Balls",
                    show_plot=False)

        if ball_number_1 == ball_number_2:
            break

    return gb_dict_new


# 去重叠逻辑 (保持不变)
def isOverlap(gb_dict):
    Flag = True
    later_dict = gb_dict.copy()
    while True:
        ball_number_1 = len(gb_dict)
        centers = []
        keys = []
        dict_overlap = {}
        center_radius = {}

        for key in later_dict.keys():
            center, radius_mean = calculate_center_and_radius(later_dict[key][0])
            center_radius[key] = [center, later_dict[key][0], later_dict[key][1], radius_mean]
            center_temp = []
            keys.append(key)
            for center_split in key.split('_'):
                center_temp.append(float(center_split))
            centers.append(center_temp)
        centers = np.array(centers)

        if Flag:
            later_dict = {}
            Flag = False

        for i, center01 in enumerate(centers):
            for j, center02 in enumerate(centers):
                if i < j and center01[0] != center02[0]:
                    dist = calculate_distances(center_radius[keys[i]][0], center_radius[keys[j]][0])
                    if dist < center_radius[keys[i]][3] + center_radius[keys[j]][3]:
                        dict_overlap[keys[i]] = center_radius[keys[i]][1:3]
                        dict_overlap[keys[j]] = center_radius[keys[j]][1:3]

        if len(dict_overlap) == 0:
            gb_dict.update(later_dict)
            ball_number_2 = len(gb_dict)
            if ball_number_1 != ball_number_2:
                Flag = True
                later_dict = gb_dict.copy()
            else:
                return gb_dict

        gb_dict_single = dict_overlap.copy()
        for i in range(len(gb_dict_single)):
            gb_single = {}
            dict_temp = gb_dict_single.popitem()
            gb_single[dict_temp[0]] = dict_temp[1]
            if len(dict_temp[1][0]) == 1:
                later_dict.update(gb_single)
                continue
            gb_dict_new = splits_ball(gb_single).copy()
            later_dict.update(gb_dict_new)


# 分裂粒球具体实现 (保持不变)
def splits_ball(gb_dict):
    center = []
    distances_other_class = []
    balls = []
    center_other_class = []
    ball_list = {}
    distances_other_temp = []
    centers_dict = []
    gbs_dict = []
    distances_dict = []

    gb_dict_temp = gb_dict.popitem()
    for center_split in gb_dict_temp[0].split('_'):
        center.append(float(center_split))
    center = np.array(center)
    gb = gb_dict_temp[1][0]
    distances = gb_dict_temp[1][1]
    centers_dict.append(center)

    len_label = np.unique(gb[:, 0], axis=0)
    if len(len_label) > 1:
        gb_class = len(len_label)
    else:
        gb_class = 2

    len_label = len_label.tolist()
    for i in range(0, gb_class - 1):
        if len(len_label) < 2:
            gb_temp = np.delete(gb, np.argmin(distances), axis=0)
            ran = random.randint(0, len(gb_temp) - 1)
            center_other_temp = gb_temp[ran]
            center_other_class.append(center_other_temp)
        else:
            if center[0] in len_label:
                len_label.remove(center[0])
            gb_temp = gb[gb[:, 0] == len_label[i], :]
            ran = random.randint(0, len(gb_temp) - 1)
            center_other_temp = gb_temp[ran]
            center_other_class.append(center_other_temp)

    centers_dict.extend(center_other_class)
    distances_other_class.append(distances)

    for center_other in center_other_class:
        distances_other = []
        for feature in gb:
            distances_other.append(calculate_distances(feature[1:], center_other[1:]))
        distances_other_temp.append(distances_other)
        distances_other_class.append(distances_other)

    for i in range(len(distances)):
        distances_temp = []
        distances_temp.append(distances[i])
        for distances_other in distances_other_temp:
            distances_temp.append(distances_other[i])
        classification = distances_temp.index(min(distances_temp))
        balls.append(classification)

    balls_array = np.array(balls)

    for i in range(0, len(centers_dict)):
        gbs_dict.append(gb[balls_array == i, :])

    i = 0
    for j in range(len(centers_dict)):
        distances_dict.append([])
    for label in balls:
        distances_dict[label].append(distances_other_class[label][i])
        i += 1

    for i in range(len(centers_dict)):
        center_array = np.array(centers_dict[i])
        gb_dict_key = str(float(center_array[0]))
        for j in range(1, len(center_array)):
            gb_dict_key += '_' + str(float(center_array[j]))
        gb_dict_value = [gbs_dict[i], distances_dict[i]]
        ball_list[gb_dict_key] = gb_dict_value

    return ball_list


# ----------------- 主程序修改部分：加载和降维 -----------------

def main():
    np.set_printoptions(suppress=True)

    # ------------------- 1. 加载高效数据集 -------------------
    print("正在加载 sklearn 自带的高效手写数字数据集...")
    digits = load_digits()

    # 集中使用数据，并进行预处理
    # 数据规模：1797个样本，64维
    data_X = digits.data.astype('float32') / 16.0  # 归一化 (0-16 -> 0.0-1.0)
    data_Y = digits.target.reshape(-1, 1)

    # ------------------- 2. 降维处理 (PCA) -------------------
    print(f"正在对 {data_X.shape[0]} 个样本进行 PCA 降维 (64D -> 2D)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data_X)

    # 构建适合 GB 算法的数据格式: [标签, 特征1 (PCA1), 特征2 (PCA2)]
    data = np.hstack((data_Y, X_pca))

    # ------------------- 3. 运行 GB 算法 -------------------

    print(f"最终用于运算的数据形状: {data.shape}")
    print("开始进行 Granular-Ball 运算...")

    start_total = time.time()
    best_ball_list = []

    # 循环运行 1 次 (现在速度很快，可按需增加)
    for i in range(1):
        start = time.time()

        # 1. 初始化
        purity_init = get_label_and_purity(data)[1]
        center_init = data[random.randint(0, len(data) - 1), :]
        distance_init = [calculate_distances(feature[1:], center_init[1:]) for feature in data]

        # 2. 封装进字典
        gb_dict = {}
        # 简化 key 的生成，只取标签和前两个 PCA 特征
        center_array = center_init.tolist()
        gb_dict_key = f"{center_array[0]:.0f}_{center_array[1]:.4f}_{center_array[2]:.4f}"
        gb_dict[gb_dict_key] = [data, distance_init]

        # 3. 分裂 (核心步骤)
        if PLOT_ITERATION:
            # 设置 Matplotlib 交互模式，用于动态绘图
            plt.ion()
            plt.figure(figsize=(10, 8))

        gb_dict = splits(purity_init=purity_init, gb_dict=gb_dict)

        if PLOT_ITERATION:
            plt.ioff()
            plt.close()  # 关闭迭代过程图

        # 4. 利用 K-Means 进行全局划分优化 (使用最终粒球数量作为 K)
        k_centers = []
        splits_k = len(gb_dict)
        # 提取中心 (现在速度快很多了，因为维度很低)
        for key in gb_dict.keys():
            k_centers.append([float(k) for k in key.split('_')][1:])

        final_ball_list = []
        if splits_k > 0:
            # 简化 k_means，让它自己寻找最佳中心，以确保稳定
            label_cluster = k_means(X=data[:, 1:], n_clusters=splits_k, n_init='auto', random_state=5)[1]
            for single_label in range(splits_k):
                final_ball_list.append(data[label_cluster == single_label, :])
        else:
            final_ball_list.append(data)

        best_ball_list = final_ball_list

        end = time.time()
        print(f"--- Run {i + 1} 耗时: {round(end - start, 2)}秒 ---")

        # 5. 最终绘图
        plot_gb(best_ball_list, title=f"Final Granular Ball Result ({len(best_ball_list)} Balls)", show_plot=True)

    end_total = time.time()
    print('-' * 40)
    print(f'✅ **程序运行成功!**')
    print(f'✅ **Granular-Balls 最终数量：{len(best_ball_list)}**')
    print(f'✅ **总耗时：{round(end_total - start_total, 2)} 秒**')

if __name__ == '__main__':
    main()
