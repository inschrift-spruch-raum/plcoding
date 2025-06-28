#include "utils.h"


NDimShape::NDimShape(const int *shape, int ndim) {
    this->shape = new int[ndim];
    this->ndim = ndim;
    this->size = 1;
    for (int i = 0; i < ndim; ++i) {
        this->shape[i] = shape[i];
        this->size *= shape[i];
    }
    this->nd_indices = new int*[this->size];
    for (int i = 0; i < this->size; ++i) {
        this->nd_indices[i] = new int[ndim];
        int rem = i;
        for (int j = ndim - 1; j >= 0; --j) {
            this->nd_indices[i][j] = rem % this->shape[j];
            rem /= this->shape[j];
        }
    }
    // fast operation
    this->array1 = fftw_alloc_real(this->size);
    this->array2 = fftw_alloc_real(this->size);
    this->array_ = fftw_alloc_real(this->size);
    this->farray1 = fftw_alloc_complex(this->size);
    this->farray2 = fftw_alloc_complex(this->size);
    this->farray_ = fftw_alloc_complex(this->size);
    this->plan1 = fftw_plan_dft_r2c(ndim, shape, array1, farray1, FFTW_MEASURE);
    this->plan2 = fftw_plan_dft_r2c(ndim, shape, array2, farray2, FFTW_MEASURE);
    this->plan_ = fftw_plan_dft_c2r(ndim, shape, farray_, array_, FFTW_MEASURE);
}

NDimShape::~NDimShape() {
    delete[] this->shape;
    for (int i = 0; i < this->size; ++i)
        delete[] this->nd_indices[i];
    delete[] this->nd_indices;
    fftw_destroy_plan(this->plan1);
    fftw_destroy_plan(this->plan2);
    fftw_destroy_plan(this->plan_);
    fftw_free(this->array1);
    fftw_free(this->array2);
    fftw_free(this->array_);
    fftw_free(this->farray1);
    fftw_free(this->farray2);
    fftw_free(this->farray_);
}

int NDimShape::to_linear(const int *index) {
    int k = 0;
    for (int i = 0; i < this->ndim; ++i)
        k = k * this->shape[i] + index[i];
    return k;
}

void NDimShape::nrmcomb(const double *from1, const double *from2, double *to) {
    for (int i = 0; i < this->size; ++i)
        to[i] = from1[i] * from2[i];
    this->normalize(to);
}

void NDimShape::circonv(const double *from1, const double *from2, double *to) {
    for (int i = 0; i < this->size; ++i) {
        this->array1[i] = from1[i];
        this->array2[i] = from2[i];
    }
    fftw_execute(this->plan1);
    fftw_execute(this->plan2);
    // multiplication of complex numbers
    for (int i = 0; i < this->size; ++i) {
        double a = this->farray1[i][0], b = this->farray1[i][1];
        double c = this->farray2[i][0], d = this->farray2[i][1];
        this->farray_[i][0] = a * c - b * d;
        this->farray_[i][1] = a * d + b * c;
    }
    fftw_execute(this->plan_);
    for (int i = 0; i < this->size; ++i)
        to[i] = this->array_[i] / this->size;
}

void NDimShape::normalize(double *target) {
    double tau = 0.0;
    for (int i = 0; i < this->size; ++i)
        tau += target[i];
    for (int i = 0; i < this->size; ++i)
        target[i] /= tau;
}

JointProb::JointProb(NDimShape *shape) {
    this->shape = shape;
    this->data = new double[shape->get_size()];
    this->values = new int[shape->get_ndim()];
    this->reset();
}

JointProb::~JointProb() {
    delete[] this->data;
    delete[] this->values;
}

void JointProb::reset() {
    for (int i = 0; i < this->shape->get_ndim(); ++i)
        this->values[i] = -1;
    this->set_uniform();
}

void JointProb::calc_marginal(int var, double *output) {
    for (int i = 0; i < this->shape->get_base(var); ++i)
        output[i] = 0.0;
    for (int i = 0; i < this->shape->get_size(); ++i) {
        int j = this->shape->from_linear(i)[var];
        output[j] += this->data[i];
    }
}

double JointProb::calc_marginal(int var, int value) {
    double result = 0.0;
    for (int i = 0; i < this->shape->get_size(); ++i)
        if (value == this->shape->from_linear(i)[var])
            result += this->data[i];
    return result;
}

void JointProb::get_value(int *output) {
    for (int i = 0; i < this->shape->get_ndim(); ++i)
        output[i] = this->values[i];
}

void JointProb::set_uniform() {
    int size = this->shape->get_size();
    for (int i = 0; i < size; ++i)
        this->data[i] = 1.0 / size;
}

void JointProb::decision_upon(int var, int value) {
    this->values[var] = value;
    for (int i = 0; i < this->shape->get_size(); ++i) {
        this->data[i] = 1.0;
        for (int j = 0; j < this->shape->get_ndim(); ++j) {
            if (this->shape->from_linear(i)[j] != this->values[j] && this->values[j] != -1) {
                this->data[i] = 0.0;
                break;
            }
        }
    }
    this->shape->normalize(this->data);
}

void JointProb::set_prior(const double *prior) {
    for (int i = 0; i < this->shape->get_size(); ++i)
        this->data[i] = prior[i];
}

void JointProb::sum(JointProb *input1, JointProb *input2) {
    for (int i = 0; i < this->shape->get_ndim(); ++i) {
        int value1 = input1->values[i];
        int value2 = input2->values[i];
        if (value1 != -1 && value2 != -1)
            this->values[i] = (value1 + value2) % this->shape->get_base(i);
        else
            this->values[i] = -1;
    }
    shape->circonv(input1->data, input2->data, this->data);
}

void JointProb::copy_from(JointProb *input) {
    for (int i = 0; i < this->shape->get_ndim(); ++i)
        this->values[i] = input->values[i];
    for (int i = 0; i < this->shape->get_size(); ++i)
        this->data[i] = input->data[i];
}

void JointProb::combine(JointProb *input1, JointProb *input2) {
    for (int i = 0; i < this->shape->get_ndim(); ++i) {
        int value1 = input1->values[i];
        int value2 = input2->values[i];
        if (value1 != value2 && value1 != -1 && value2 != -1)
            throw std::runtime_error("Invalid probs for JointProb::combine()!");
        this->values[i] = (value2 == -1) ? value1 : value2;
    }
    this->shape->nrmcomb(input1->data, input2->data, this->data);
}

void JointProb::reverse(JointProb *input) {
    auto shape = this->shape;
    // let's borrow the allocated values' space
    for (int i = 0; i < shape->get_size(); ++i) {
        const int *nd_index = shape->from_linear(i);
        for (int j = 0; j < shape->get_ndim(); ++j) {
            int base = shape->get_base(j);
            this->values[j] = (base - nd_index[j]) % base;
        }
        int k = shape->to_linear(this->values);
        this->data[k] = input->data[i];
    }
    // calculate the reversed values
    for (int i = 0; i < shape->get_ndim(); ++i) {
        int value = input->values[i];
        this->values[i] = value;
        if (value != -1) {
            int base = shape->get_base(i);
            this->values[i] = (base - value) % base;
        }
    }
}

void JointProb::subtract(JointProb *input1, JointProb *input2) {
    this->reverse(input2);
    this->sum(input1, this);
}

void JointProb::print(bool _not_last) {
    int size = this->shape->get_size();
    int ndim = this->shape->get_ndim();
    std::cout << "(";
    for (int i = 0; i < size - 1; ++i)
        std::cout << std::fixed << std::setprecision(2) << this->data[i] << ", ";
    std::cout << std::fixed << std::setprecision(2) << this->data[size - 1] << ")<";
    for (int i = 0; i < ndim - 1; ++i)
        std::cout << this->values[i] << ", ";
    std::cout << this->values[ndim - 1] << ">";
    if (_not_last) std::cout << ", ";
}

Edge::Edge(int size, NDimShape *shape) {
    this->size = size;
    this->probs = new JointProb*[size];
    for (int i = 0; i < size; ++i)
        this->probs[i] = new JointProb(shape);
    this->reset();
}

Edge::~Edge() {
    for (int i = 0; i < size; ++i)
        delete this->probs[i];
    delete[] this->probs;
}

void Edge::reset() {
    for (int i = 0; i < this->size; ++i)
        this->probs[i]->reset();
    this->node_from = nullptr;
}

void Edge::set_probs(const double *probs) {
    int stride = this->get_prob()->get_shape()->get_size();
    for (int i = 0; i < this->size; ++i)
        this->probs[i]->set_prior(probs + i * stride);
}

void Edge::copy_from(const Edge *input) {
    for (int i = 0; i < this->size; ++i)
        this->probs[i]->copy_from(input->probs[i]);
    this->node_from = input->node_from;
}

void Edge::combine_with(const Edge *input) {
    for (int i = 0; i < this->size; ++i)
        this->probs[i]->combine(this->probs[i], input->probs[i]);
}

void Edge::print(bool _not_last) {
    std::cout << "{";
    for (int j = 0; j < this->size - 1; ++j)
        this->probs[j]->print();
    this->probs[this->size - 1]->print(false);
    std::cout << "}";
    if (_not_last) std::cout << ", ";
}

void Edge::update_root(Edge *root, Edge *lchl, Edge *rchl) {
    int size = lchl->size;
    for (int i = 0; i < size; ++i) {
        root->probs[i]->sum(lchl->probs[i], rchl->probs[i]);
        root->probs[i + size]->copy_from(rchl->probs[i]);
    }
}

void Edge::update_lchl(Edge *root, Edge *lchl, Edge *rchl) {
    int size = lchl->size;
    JointProb temp = JointProb(lchl->get_prob()->get_shape());
    for (int i = 0; i < size; ++i) {
        temp.combine(rchl->probs[i], root->probs[i + size]);
        lchl->probs[i]->subtract(root->probs[i], &temp);
    }
}

void Edge::update_rchl(Edge *root, Edge *lchl, Edge *rchl) {
    int size = lchl->size;
    JointProb temp = JointProb(lchl->get_prob()->get_shape());
    for (int i = 0; i < size; ++i) {
        temp.subtract(root->probs[i], lchl->probs[i]);
        rchl->probs[i]->combine(&temp, root->probs[i + size]);
    }
}

Node::Node(int branch) {
    this->branch = branch;
    this->reset();
}

void Node::reset() {
    this->edge_root = nullptr;
    this->edge_lchl = nullptr;
    this->edge_rchl = nullptr;
}

void Node::copy_from(const Node *input) {
    this->branch = input->branch;
    this->edge_root = input->edge_root;
    this->edge_lchl = input->edge_lchl;
    this->edge_rchl = input->edge_rchl;
}

Edge *Node::get_ptr(int type) {
    if (type == 0) {
        return this->edge_root;
    } else if (type == 1) {
        return this->edge_lchl;
    } else if (type == 2) {
        return this->edge_rchl;
    } else {
        throw std::runtime_error("Invalid type for Node::get_ptr()!");
    }
}

void Node::set_ptr(int type, Edge *edge) {
    if (type == 0) {
        this->edge_root = edge;
    } else if (type == 1) {
        this->edge_lchl = edge;
    } else if (type == 2) {
        this->edge_rchl = edge;
    } else {
        throw std::runtime_error("Invalid type for Node::set_ptr()!");
    }
}

void Node::update_edge(int type) {
    if (type == 0) {
        Edge::update_root(this->edge_root, this->edge_lchl, this->edge_rchl);
    } else if (type == 1) {
        Edge::update_lchl(this->edge_root, this->edge_lchl, this->edge_rchl);
    } else if (type == 2) {
        Edge::update_rchl(this->edge_root, this->edge_lchl, this->edge_rchl);
    } else {
        throw std::runtime_error("Invalid type for Node::set_ptr()!");
    }
}

int Node::eval_relation(int branch_from, int branch_to) {
    if (branch_from == branch_to) {
        return -1;
    } else if (branch_to == branch_from * 2 + 1) {
        return 1;
    } else if (branch_to == branch_from * 2 + 2) {
        return 2;
    } else if (branch_to == (branch_from - 1) / 2) {
        return 0;
    } else {
        return -1;
    }
}

Tree::Tree(int n_level, NDimShape *shape) {
    this->n_level = n_level;
    // initialize edges
    this->edge_num = (1 << n_level) - 1;
    this->edges = new Edge*[this->edge_num];
    for (int i = 0; i < this->edge_num; ++i)
        this->edges[i] = new Edge(this->edge_size(Tree::get_depth(i)), shape);
    // initialize nodes
    this->node_num = (1 << (n_level - 1)) - 1;
    this->nodes = new Node*[this->node_num];
    for (int i = 0; i < this->node_num; ++i)
        this->nodes[i] = new Node(i);
    this->reset();
}

Tree::~Tree() {
    for (int i = 0; i < this->edge_num; ++i)
        delete this->edges[i];
    delete[] this->edges;
    for (int i = 0; i < this->node_num; ++i)
        delete this->nodes[i];
    delete[] this->nodes;
}

void Tree::reset() {
    // reset the nodes
    for (int i = 0; i < this->node_num; ++i) {
        this->nodes[i]->set_ptr(0, this->edges[i]);
        this->nodes[i]->set_ptr(1, this->edges[i * 2 + 1]);
        this->nodes[i]->set_ptr(2, this->edges[i * 2 + 2]);
    }
    // reset the edges
    this->edges[0]->set_node_from(nullptr);
    for (int i = 1; i < this->node_num; ++i) {
        this->edges[i]->reset();
        this->edges[i]->set_node_from(this->nodes[i]);
    }
    for (int i = this->node_num; i < this->edge_num; ++i)
        this->edges[i]->reset();
}

void Tree::set_root(const double *probs) {
    this->edges[0]->set_probs(probs);
}

void Tree::print() {
    for (int i = 0; i < this->edge_num; ++i) {
        if (Tree::get_depth(i + 1) != Tree::get_depth(i)) {
            this->edges[i]->print(false);
            std::cout << std::endl;
        } else
            this->edges[i]->print();
    }
}

Walker::Walker(int code_len, NDimShape *shape) {
    this->code_lvl = std::ceil(std::log2(code_len));
    this->code_len = (1 << this->code_lvl);
    this->tree = new Tree(this->code_lvl + 1, shape);
    this->edge_buffers = new Edge*[this->code_lvl + 1];
    for (int i = 0; i < code_lvl + 1; ++i)
        this->edge_buffers[i] = new Edge(this->tree->edge_size(i), shape);
    this->node_buffer = new Node(-1);
    this->reset();
}

Walker::~Walker() {
    delete this->tree;
    for (int i = 0; i < code_lvl + 1; ++i)
        delete this->edge_buffers[i];
    delete[] this->edge_buffers;
    delete this->node_buffer;
}

void Walker::reset() {
    this->tree->reset();
    this->head = this->tree->get_node(0);
}

void Walker::set_priors(const double *priors) {
    this->tree->set_root(priors);
}

void Walker::lazy_step(int branch_to) {
    int branch_now = this->head->get_branch();
    int type = Node::eval_relation(branch_now, branch_to);
    // get the corresponding edges
    int edge_branch = (type == 0) ? branch_now : branch_to;
    Edge *edge_buffer = this->get_edge_buffer(edge_branch);
    // update the data part of edge buffer (using node_buffer)
    this->node_buffer->copy_from(this->head);
    this->node_buffer->set_ptr(type, edge_buffer);
    this->node_buffer->update_edge(type);
    // update the relation part of edge buffer
    edge_buffer->set_node_from(this->head);
    // update the relation part of node buffer
    Node *node_prev = this->head->get_ptr(type)->get_node_from();
    Edge *edge_real = this->tree->get_edge(edge_branch);
    int type_rev = Node::eval_relation(branch_to, branch_now);
    // update the node buffer
    this->node_buffer->copy_from(node_prev);
    this->node_buffer->set_ptr(type_rev, edge_real);
}

void Walker::flush() {
    int branch_now = this->head->get_branch();
    int branch_to  = this->node_buffer->get_branch();
    int type = Node::eval_relation(branch_now, branch_to);
    // copy the buffer data into real memory
    int edge_branch = (type == 0) ? branch_now : branch_to;
    Edge *edge_real = this->get_edge_real(edge_branch);
    Edge *edge_buffer = this->get_edge_buffer(edge_branch);
    edge_real->copy_from(edge_buffer);
    // and then step into the corresponding node
    Node *node_real = this->get_node_real(branch_to);
    node_real->copy_from(this->node_buffer);
    this->head = node_real;
}

void Walker::print_tree() {
    std::cout << "walker's current branch=" << this->head->get_branch() << ", with tree:\n";
    this->tree->print();
}

JointProb *Walker::calc_leaf(int index) {
    int type = (index % 2 == 0) ? 1 : 2;
    // calculate the joint distribution
    Edge *edge_buffer = this->edge_buffers[this->code_lvl];
    this->node_buffer->copy_from(this->head);
    this->node_buffer->set_ptr(type, edge_buffer);
    this->node_buffer->update_edge(type);
    // and combine it with previous decision
    Edge *edge_prev = this->head->get_ptr(type);
    edge_buffer->combine_with(edge_prev);
    return edge_buffer->get_prob();
}

// get the path between any two branches on the decoding tree
int *Walker::get_path(int branch_from, int branch_to) {
    // allocate memory for branches
    int path_len = Tree::get_depth(branch_from) + Tree::get_depth(branch_to) + 1;
    int *path = new int[path_len];
    // the computation proceeds to their common root node
    int up = 0, dn = path_len - 1;
    while (branch_from != branch_to) {
        while (branch_from > branch_to) {
            path[up] = branch_from;
            branch_from = (branch_from - 1) / 2;
            up += 1;
        }
        while (branch_from < branch_to) {
            path[dn] = branch_to;
            branch_to = (branch_to - 1) / 2;
            dn -= 1;
        }
    }
    // save their common root
    path[dn] = branch_to;
    // and fill the remaining positions with -1 
    for (int i = up; i < dn; ++i)
        path[i] = -1;
    return path;
}

// lead the walkers going through the path of the decoding tree
void Walker::walk_to(Walker **walkers, int n_walker, int branch_to) {
    int *path = Walker::get_path(walkers[0]->head->get_branch(), branch_to);
    int branch_now = path[0], *branch_next = path + 1;
    while (branch_now != branch_to) {
        if (*branch_next != -1) {
            if (branch_now != -1) {
                for (int i = 0; i < n_walker; ++i)
                    walkers[i]->lazy_step(*branch_next);
                for (int i = 0; i < n_walker; ++i)
                    walkers[i]->flush();
            }
            branch_now = *branch_next;
        }
        branch_next += 1;
    }
    delete[] path;
}
