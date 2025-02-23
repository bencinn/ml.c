double sigmoid(double value);

typedef struct Layer {
    int layer_size;
    double* neuron_weight;
    double* neuron_bias;
    struct Layer* next;
} Layer;

Layer* new_layer(int layer_size);
void layer_new_next(Layer* l, int next_size);

double* forward_props(Layer* l, double* input);
double* full_forward_props(Layer* l, double* input);
void free_all_layer_after(Layer* l);
