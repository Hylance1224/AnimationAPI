import AnalyzeNode
# 退出条件


def is_similar_ids(ids1, ids2):
    if len(ids1) != len(ids2):
        return False
    else:
        for i in range(len(ids2)):
            if ids1[i] != ids2[i]:
                return False
    return True


# 判断是否长时间一直处于同一个页面
def is_same_page(global_num, store_path, threshold = 10):
    if global_num > threshold + 1:
        previous_ids = AnalyzeNode.get_all_ids(store_path + str(global_num - threshold - 1) + '.dom')
        for i in range(global_num - threshold, global_num):
            ids = AnalyzeNode.get_all_ids(store_path + str(i) + '.dom')
            if not is_similar_ids(previous_ids, ids):
                return False
    else:
        return False
    return True


# 检查是否连续10次操作都在同一个activity
def is_same_activity(global_num, store_path, threshold = 20):
    if global_num > threshold + 1:
        content0 = open(store_path + str(global_num - threshold - 1) + 'operate.txt', "r", encoding='utf-8').read()
        lines0 = content0.splitlines(False)
        for i in range(global_num - threshold + 1, global_num + 1):
            f = open(store_path + str(i) + 'operate.txt', "r", encoding='utf-8')
            lines = f.read().splitlines(False)
            if len(lines) > 7:
                if lines[7] != lines0[7]:
                    return False
    else:
        return False
    return True



# 检查是否长期点击空白
def is_long_blank(global_num, store_path, threshold = 2):
    if global_num > threshold + 1:
        for i in range(global_num - threshold + 1, global_num + 1):
            f = open(store_path + str(i) + 'operate.txt', "r", encoding='utf-8')
            lines = f.read().splitlines(False)
            if lines[0] != 'blank':
                return False
    else:
        return False
    return True


def have_lot_blank(global_num, store_path):
    num = 0
    for i in range(1, global_num):
        f = open(store_path + str(i) + 'operate.txt', "r", encoding='utf-8')
        lines = f.read().splitlines(False)
        if lines[0] == 'blank':
            num = num + 1
    if global_num > 100 and num/global_num>0.6:
        return True
    return False


def need_restart(global_num, store_path, global_d):
    if is_same_page(global_num, store_path):
        # print('长期处于同一页面')
        return True
    elif is_same_activity(global_num, store_path):
        # print('长期处于同一activity')
        return True
    elif is_long_blank(global_num, store_path):
        # print('长期点击空白')
        return True
    elif 'AdActivity' in global_d.app_current()['activity']:
        return True
    return False


def need_stop(global_num, store_path):
    if global_num > 500:
        return True
    elif have_lot_blank(global_num, store_path):
        print('有太多空白页面，退出')
        return True
    elif is_long_blank(global_num, store_path, 10):
        print('连续10次空白页，退出')
        return True
    elif global_num == 15 and is_same_activity(global_num, store_path, 12):
        print('只有一个activity, 退出')
        return True
    elif is_same_activity(global_num, store_path, 40):
        print('长期处于同一activity, 退出')
        return True
    elif is_same_page(global_num, store_path, 50):
        print('太多相同页面，退出')
        return True


if __name__ == '__main__':
    t = is_same_page(13, 'G:/output_animation/photo/7/')
    print(t)