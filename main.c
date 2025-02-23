#include<stdio.h>
#include<stdbool.h>
#include<assert.h>
#include "ml.h"
#include "math.h"

#define new_test(s,d) {printf("[testing] %s\n", s); bool out=d; if(!out) printf("[failed test] %s\n", s); else printf("[test success] %s\n", s);}
#define asas(a) if(!(a)) return false;
#define asnear(a,b) if (fabs(b-a)>0.0005) return false;

bool test_new_layers() {
    Layer* l = new_layer(3);
    layer_new_next(l, 2);
    layer_new_next(l, 4);
    layer_new_next(l,5);
    asas(l->layer_size==3);
    asas(l->next->layer_size==2);
    asas(l->next->next->layer_size==4);
    asas(l->next->next->next->layer_size==5);
    asas(l->next->next->next->next==NULL);
    free_all_layer_after(l);
    return true;
}

bool test_sigmoid() {
    asnear(sigmoid(0.4), 0.5986876601);
    return true;
}

bool test_forward_props_two_layer() {
    Layer* l = new_layer(2);
    layer_new_next(l, 3);
    assert(l->neuron_weight!=NULL);
    // setting weight and bias
    l->neuron_weight[0]=0.5;
    l->neuron_weight[1]=0.4;
    l->neuron_weight[2]=0.3;
    l->neuron_weight[3]=0.2;
    l->neuron_weight[4]=0.1;
    l->neuron_weight[5]=0.0;
    l->next->neuron_bias[0]=0.3;
    l->next->neuron_bias[1]=0.2;
    l->next->neuron_bias[2]=0.1;
    double d[2] = {0.8, 0.2};
    double* test = forward_props(l, d);
    asnear(test[0], 0.67699);
    asnear(test[1], 0.63181);
    asnear(test[2], 0.58419);
    free_all_layer_after(l);
    return true;
}

bool test_forward_props_multi_layer() {
    Layer* l = new_layer(2);
    layer_new_next(l, 3);
    layer_new_next(l, 2);
    assert(l->neuron_weight!=NULL);
    // setting weight and bias
    l->neuron_weight[0]=0.5;
    l->neuron_weight[1]=0.4;
    l->neuron_weight[2]=0.3;
    l->neuron_weight[3]=0.2;
    l->neuron_weight[4]=0.1;
    l->neuron_weight[5]=0.0;
    l->next->neuron_bias[0]=0.3;
    l->next->neuron_bias[1]=0.2;
    l->next->neuron_bias[2]=0.1;
    l->next->neuron_weight[0]=0.1;
    l->next->neuron_weight[1]=0.2;
    l->next->neuron_weight[2]=0.3;
    l->next->neuron_weight[3]=0.4;
    l->next->neuron_weight[4]=0.5;
    l->next->neuron_weight[5]=0.6;
    l->next->next->neuron_bias[0]=0.2;
    l->next->next->neuron_bias[1]=0.1;
    double d[2] = {0.8, 0.2};
    double* test = full_forward_props(l, d);
    asnear(test[0], 0.67903);
    asnear(test[1], 0.69817);
    free_all_layer_after(l);
    return true;
}

int main() {
    printf("test ml\n");
    new_test("test creating new layer", test_new_layers());
    new_test("test sigmoid", test_sigmoid());
    new_test("test forward props (two layer)", test_forward_props_two_layer());
    new_test("test forward props (multi layer)", test_forward_props_multi_layer());
}
