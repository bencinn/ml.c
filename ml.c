#include "ml.h"
#include<stdlib.h>
#include<math.h>
#include<stdio.h>

// https://stackoverflow.com/a/36890692
#define MAT(arr,i,j,M) arr[i*M+j]

// Logistic function sigmoid
// TODO: implement more sigmoid for fun and benchmark
double sigmoid(double value) {
    return 1/(1+exp(-value));
}

Layer* new_layer(int layer_size) {
    Layer* l = malloc(sizeof(Layer));
    l->layer_size=layer_size;
    l->next=NULL;
    return l;
}

void layer_new_next(Layer* l, int next_size) {
    while(l->next!=NULL) l=l->next;
    Layer* next = new_layer(next_size);
    l->next=next;
    l->neuron_weight=malloc(next_size*l->layer_size*sizeof(double));
    next->neuron_bias=malloc(sizeof(double)*next_size);
}

void free_all_layer_after(Layer* l) {
    if(l->next!=NULL) {
        free_all_layer_after(l->next);
    }
    free(l->neuron_weight);
    free(l->neuron_bias);
    free(l);
}

// FOR GOD FUCKING SAKE PLEASE FREE 
double* forward_props(Layer* l, double* input) {
    if (l->next==NULL) return NULL;
    int ns = l->next->layer_size;
    double* next_input = malloc(sizeof(double)*ns);
    for(int i=0;i<ns;i++) {
        double inner = 0;
        for(int j=0;j<l->layer_size;j++) {
            inner += input[j]*MAT(l->neuron_weight,j,i,ns);
        }
        inner += l->next->neuron_bias[i];
        next_input[i]=sigmoid(inner);
    }
    return next_input;
}

double* full_forward_props(Layer* l, double* input) {
    double* out = forward_props(l, input);
    double* old_out;
    l=l->next;
    while(l->next!=NULL) {
        old_out = out;
        out=forward_props(l, old_out);
        free(old_out);
        l=l->next;
    }
    return out;
}

double error_calc(double* a, double* b, int n) {
    double cumulative = 0;
    for(int i=0;i<n;i++) {
        cumulative+=pow((a[i]-b[i]),2);
    }
    return sqrt(cumulative/n);
}
