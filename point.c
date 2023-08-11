#include "point.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void computeCoordinates(Point* point, double t) {
    /*
    Calculate the coordinates (x, y) of the point P using the given formula.
    Parameters:
        point: Pointer to the Point structure.
        t: Value of parameter t for coordinate computation.
    */

    // Calculate x coordinate
    point->x = ((point->x2 - point->x1) / 2) * sin(t * M_PI / 2) + (point->x2 + point->x1) / 2;

    // Calculate y coordinate
    point->y = point->a * point->x + point->b;
}
