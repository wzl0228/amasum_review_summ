import json

# max_src_ntokens_per_sent

# list1 = json.load(open('test_re_len_list.json', 'r', encoding='utf-8'))
# print(max(list1)) # 570
# print(min(list1)) # 10
#
# list2 = json.load(open('train_re_len_list.json', 'r', encoding='utf-8'))
# print(max(list2)) # 621
# print(min(list2)) # 10
#
# list3 = json.load(open('valid_re_len_list.json', 'r', encoding='utf-8'))
# print(max(list3)) # 334
# print(min(list3)) # 10
#
# all_list = list1 + list2 + list3
# sorted_all_list = sorted(all_list)
# boundary_idx = int(len(sorted_all_list) * 0.985) # 0.90->126 0.95->139 0.985->150 0.99->154 0.999->180
# boundary = sorted_all_list[boundary_idx]
# print("max_src_ntokens_per_sent值为：", boundary) # 150

# max_tgt_len
list1 = json.load(open('test_summ_len_list.json', 'r', encoding='utf-8'))
print(max(list1)) # 481
print(min(list1)) # 26

list2 = json.load(open('train_summ_len_list.json', 'r', encoding='utf-8'))
print(max(list2)) # 688
print(min(list2)) # 28

list3 = json.load(open('valid_summ_len_list.json', 'r', encoding='utf-8'))
print(max(list3)) # 439
print(min(list3)) # 27

all_list = list1 + list2 + list3
sorted_all_list = sorted(all_list)
boundary_idx = int(len(sorted_all_list) * 0.985) # 0.90->127 0.95->175 0.97->250 0.985->303 0.99->326 0.999->436
boundary = sorted_all_list[boundary_idx]
print("max_tgt_len值为：", boundary) # 303