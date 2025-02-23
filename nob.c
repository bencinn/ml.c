// nob.c
#define NOB_IMPLEMENTATION
#include "nob.h"

int main(int argc, char **argv)
{
    NOB_GO_REBUILD_URSELF(argc, argv);
    Nob_Cmd build_main = {0};
    nob_cmd_append(&build_main, "cc", "-g", "-Wall", "-Wextra", "-o", "main", "main.c", "ml.c", "-lm");
    if (!nob_cmd_run_sync(build_main)) return 1;
    return 0;
}
