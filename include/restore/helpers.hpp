#ifndef RESTORE_HELPERS_H
#define RESTORE_HELPERS_H

#define UNUSED(expr) (void)(expr)

#ifdef BACKWARD_CXX11
void print_stacktrace() {
    backward::StackTrace stacktrace;
    backward::Printer    printer;
    stacktrace.load_here(32);
    printer.print(stacktrace);
}
#endif

#endif