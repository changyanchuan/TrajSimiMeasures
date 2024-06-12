
#include <assert.h>
#include "sfc.h"

// the bits is generally the half of bits used 
__uint128_t SFC_Z(long x, long y, const long& bits) {
    // assert(bits > 0 && x > 0 && y > 0);

	__uint128_t z = 0;
	for(long i = 0; i < bits; ++i) 
	{
        z += (x%2) * ((__uint128_t)2<<(2*i)) + ((y%2) * (__uint128_t)2<<(2*i+1));
        x /= 2;
        y /= 2;
	}
	return z;
}
