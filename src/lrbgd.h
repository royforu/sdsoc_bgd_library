
typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef union _float32_t
{
    float float32;
    unsigned int uint32;
} float32_t;




#define NUM_CHUNKS                  (4000)
#define NUM_CLASSES                 (10)
#define NUM_FEATURES                (784)

#define NUM_PARALLELERS             (1)
#define NUM_PIPEFACTOR              (5)
#if (NUM_PIPEFACTOR==5)
#define NUM_KERNEL                  (5)
#define NUM_PADDING                 (0)
#define NUM_TOTAL_FEATURE           (NUM_FEATURES + 1 + NUM_PADDING)
#else
#define NUM_KERNEL                  (2)
#define NUM_PADDING                 (5)
#define NUM_TOTAL_FEATURE           (NUM_FEATURES + 1 + NUM_PADDING)
#endif


void bgd_func0(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient);
void bgd_func1(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient);
#if (NUM_KERNEL==5)
void bgd_func2(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient);
void bgd_func3(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient);
void bgd_func4(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient);
#endif
