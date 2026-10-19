/* Deliberately ignores TERM to exercise owned simulator shutdown on failure.
   No model, GPU, network or child processes. The sleep is bounded to 60 seconds. */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc, char **argv) {
    if (argc!=2 || signal(SIGTERM,SIG_IGN)==SIG_ERR) return 2;
    FILE *marker=fopen(argv[1],"wx");
    if (!marker) return 3;
    if (fprintf(marker,"%d\n",getpid())<0 || fclose(marker)!=0) return 4;
    for (unsigned i=0;i<60;i++) sleep(1);
    return 0;
}
