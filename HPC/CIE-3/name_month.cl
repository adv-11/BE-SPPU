// Save this as name_month.cl
__kernel void print_name_month(__global const char* name, __global const char* month) {
    int gid = get_global_id(0);
    printf("Global ID: %d - Name: %s, Month: %s\n", gid, name, month);
}
