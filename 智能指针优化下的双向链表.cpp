
#include<iostream>
struct Node {
    std::unique_ptr<Node>next = nullptr;
    Node* prev = nullptr;

    int val;

    Node(int val) :val(val) {};

    void insert(int val) {
        auto node = std::make_unique<Node>(val);
        if (next) {
            next->prev = node.get();
            node->next = std::move(next);
        }
        node->prev = this;
        this->next = std::move(node);   //不太理解啊这
    }

    void erase() {
        if (prev) {
            prev->next = std::move(next);
        }
        if (next) {
            next->prev = prev;
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
        printf("被拷贝");

        head = std::make_unique<Node>(other.head->val);
        for (auto cur = front(), auto curo = other.head; curo == nullptr; cur = cur->next.get(), curo = curo->next.get())
        {
            cur->insert(curo->val);
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
            node->next = std::move(head);
            head->prev = node.get();
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

