
#include<iostream>
struct Node {
    std::unique_ptr<Node>next = nullptr;
    Node* prev = nullptr;
    int val;

    Node(int val) :val(val) {};

    void insert(int val) {
        auto node = std::make_unique<Node>(val);
        /*
        auto node = ...
        node 是这个智能指针的变量名，它的类型是 std::unique_ptr<Node>（可以理解为 “指向 Node 的独占智能指针”）。
        它的行为类似指针：
        用 node->val 访问 Node 结构体的成员（和 Node* 用法一样）。
        用 node.get() 可以获取它内部管理的原始指针（Node* 类型），但一般不建议直接操作原始指针。
        */
        if (next) {
            next->prev = node.get();
            node->next = std::move(next);
        }
        node->prev = this;
        this->next = std::move(node);   //不太理解啊这
    }

    //这里指向node->erase，进行删除操作
    void erase() {
         Node* cur = this;
 
         if (cur->prev) {
             cur->prev->next = std::move(next);
         }
         if (cur->next) {
             cur->next->prev = cur->prev;
         }
}

    ~Node() {
        printf("调用Node");
    }
};
struct List {
    std::unique_ptr<Node>head = nullptr;

    List() = default;

    List(const List& other) {
        if (!other.head) {
            head = nullptr;
            return;
        }
        printf("被拷贝");

        head= std::make_unique<Node>(other.head->val);    //定义头点
        Node* newCur = head.get();    //临时作为新的迭代点
        
        auto curo = other.head->next.get();
        while (curo) {
            newCur->insert(curo->val);
            newCur = newCur->next.get();    //要得到的是下一个点的指针位置
            curo = curo->next.get();
        }
        
    }
    List& operator=(List const&) = delete;
    List(List&&) = default;
    List& operator=(List&&) = default;

    Node* front() const {
        return head.get();
    }

    int pop_front() {
        int ret = head->val;
        head = std::move(head->next);
        return ret;
    }

    void push_front(int value) {
        auto node = std::make_unique<Node>(value);
        if (head) {
            // 1. 先让原头节点的prev指向新节点（此时head还未被移动，有效）
            head->prev = node.get();
            // 2. 再将原头节点的所有权转移给新节点的next
            node->next = std::move(head);
        }
        head = std::move(node);
    }

    Node* at(size_t index) const {
        auto curr = front();
        for (size_t i = 0; i < index; i++) {
            curr = curr->next.get();
        }
        return curr;
    }
};
void print(const List& lst) {  // 有什么值得改进的？
    printf("[");
    for (auto curr = lst.front(); curr; curr = curr->next.get()) {
        printf(" %d", curr->val);
    }
    printf(" ]\n");
}
int main() {
    List a;

    a.push_front(7);
    a.push_front(5);
    a.push_front(8);
    a.push_front(2);
    a.push_front(9);
    a.push_front(4);
    a.push_front(1);

    print(a);   // [ 1 4 9 2 8 5 7 ]

    a.at(2)->erase();

    print(a);   // [ 1 4 2 8 5 7 ]

    List b = a;

    a.at(3)->erase();

    print(a);   // [ 1 4 2 5 7 ]
    print(b);   // [ 1 4 2 8 5 7 ]

    b = {};
    a = {};

    return 0;
}

