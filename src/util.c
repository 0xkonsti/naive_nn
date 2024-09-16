#include "util.h"
#include <stdlib.h>

double randrange(double min, double max) {
    return (max - min) * ((double)rand() / (double)RAND_MAX) + min;
}
