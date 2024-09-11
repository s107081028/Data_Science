import sys
import time
class Node:
    def __init__(self, item='', value = 1, parent_node=None):
        self.item = item
        self.value = value
        self.parent = parent_node
        self.children = []
        self.link = None

    def checkifexist(self, item):  ## check if current node exist in children
        for child in self.children:
            if item == child.item:
                child.value += 1
                return child
        return None

def create_tree(database, sorted_header_table):
    root = Node(item = 'X', value = 0)
    for data in database:
        ## for every transaction, cur = root at first
        curNode = root

        for index in data:
            ## for every instance in a transaction, check if this node exists
            NextNode = curNode.checkifexist(index)
            if NextNode != None:
                curNode = NextNode
                continue

            ## if not exist, create node
            newNode = Node(item=index, parent_node=curNode)
            curNode.children.append(newNode)

            ## link same item
            if sorted_header_table[index]['head'] == None:
                sorted_header_table[index]['head'] = newNode
            else:
                linkNode = sorted_header_table[index]['head']
                while(linkNode.link != None):
                    linkNode = linkNode.link
                linkNode.link = newNode

            curNode = newNode
    
    return root, sorted_header_table

def find_path(suffix, head_node, min_sup):
    dataset = []
    header_table = {}
    # Find through the links of the header table
    while head_node != None:
        Path = []
        cur_node = head_node
        count = cur_node.value             # the value is the sum of its children
        while cur_node.parent.item != 'X':
            cur_node = cur_node.parent
            item = cur_node.item
            Path.append(item)
            if item in header_table.keys():
                header_table[item]['frequency'] += count
            else:
                header_table[item] = {}
                header_table[item]['frequency'] = count
        Path.reverse()
        Path = list(set(Path))
    # Generate dataset for constructing tree
        if len(Path) > 0:
            while count:
                dataset.append(Path)
                count -= 1
        head_node = head_node.link
    sorted_ = sorted(header_table.items(), key=lambda item: item[1]['frequency'], reverse=True)
    sorted_header_table = {}
    for key, value in sorted_:
        if value['frequency'] >= min_sup:
            sorted_header_table[key] = value
            sorted_header_table[key]['head'] = None
    freq_database = []
    for transaction in dataset:
        tx = [instance for instance in transaction if instance in sorted_header_table.keys()]
        if len(tx) > 0:
            freq_database.append(sorted(tx, key=lambda item: (sorted_header_table[item]['frequency'], item), reverse=True))

    return freq_database, sorted_header_table


def mine_tree(root, header_table, min_sup, freq_set, frequent_patterns):
    # from the least frequent item
    suffixs = [suf for suf in sorted(header_table.keys(), key=lambda item: header_table[item]['frequency'])]
    for suffix in suffixs:
        freq_temp_set = freq_set.copy()
        freq_temp_set.add(suffix)
        frequent_patterns.append((freq_temp_set, header_table[suffix]['frequency']))
        conditional_dataset, conditional_header_table = find_path(suffix, header_table[suffix]['head'], min_sup)
        conditional_tree, conditional_header_table = create_tree(conditional_dataset, conditional_header_table)
        # Recursively mining
        if conditional_header_table != {}:
            mine_tree(conditional_tree, conditional_header_table, min_sup, freq_temp_set, frequent_patterns)    

if __name__ == "__main__":
    min_sup, input_name, output_name = float(sys.argv[1]), sys.argv[2], sys.argv[3]
    start = time.time()
    ## Calculate Frequency
    header_table = {}
    num = 0
    with open(input_name, 'r') as f:
        for line in f.readlines():
            line = line.split('\n')[0]
            num += 1
            for i in line.split(','):
                if i in header_table.keys():
                    header_table[i]['frequency'] += 1
                else:
                    header_table[i] = {}
                    header_table[i]['frequency'] = 1

    ## Pop out below minimum support
    sorted_ = sorted(header_table.items(), key=lambda item: item[1]['frequency'], reverse=True)
    sorted_header_table = {}
    for key, value in sorted_:
        if value['frequency'] >= num * float(min_sup):
            sorted_header_table[key] = value
            sorted_header_table[key]['head'] = None

    min_sup = num * float(min_sup)
    ## Sort Transaction with Frequency
    database = []
    with open(input_name, 'r') as f:
        for line in f.readlines():
            line = (line.split('\n')[0]).split(',')
            tx = [instance for instance in sorted_header_table.keys() if instance in line]
            database.append(sorted(tx, key=lambda item: (sorted_header_table[item]['frequency'], item), reverse=True))

    ## Construct Tree
    root, sorted_header_table = create_tree(database, sorted_header_table)

    ## Mine Tree
    frequent_patterns = []
    mine_tree(root, sorted_header_table, min_sup, set(), frequent_patterns)

    
    # print(f'Time Consumption: {time.time() - start}')
    frequent_patterns = list(frequent_patterns)
    frequent_pattern = sorted(frequent_patterns, key=lambda item: (len(item[0]), int(sorted(list(item[0]))[0])))
    # for (i, j) in frequent_pattern:
    #     i = sorted(i, key=lambda item: int(item))
    #     print(f'{i}: {j/num}')

    with open(output_name, "w") as f:
        lines = []
        for freq_set, support in frequent_pattern:
            freq_set = sorted(list(freq_set))
            freq_set = [str(i) for i in freq_set]
            lines.append(','.join(freq_set) + ":"+format(round(support/num,4), '0.4f') + '\n')
        f.writelines(lines)

