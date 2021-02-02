#ifndef RESTORE_HELPERS_H
#define RESTORE_HELPERS_H

#define UNUSED(expr) (void)(expr)

#ifdef BACKWARD_CXX11
void print_stacktrace() {
    backward::StackTrace st;
    backward::Printer    p;
    st.load_here(32);
    p.print(st);
}
#endif

#endif