#ifndef __DPX_BOOST_WARNINGS
# ifdef __GNUC__
#  define MAC_OS_X_VERSION_MIN_REQUIRED 0
#  ifndef __cpp_noexcept_function_type
#    define __cpp_noexcept_function_type 0
#  endif
#  ifndef __NVCC___WORKAROUND_GUARD
#    define __NVCC___WORKAROUND_GUARD 0
#    define __NVCC__ 0
#  endif
#  ifndef __clang__
#    define __clang_major__ 0
#    define __clang_major___WORKAROUND_GUARD 0
#    if(__GNUC__ == 7) || (__GNUC__ == 8 && __GNUC_MINOR__ <= 3)
#      pragma GCC diagnostic push
#      pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#      pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#      pragma GCC diagnostic ignored "-Wmisleading-indentation"
#      pragma GCC diagnostic ignored "-Wcast-function-type"
#      pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#      pragma GCC diagnostic ignored "-Wparentheses"
#    endif
#  endif
#  ifdef __clang__
#    define __clang_major___WORKAROUND_GUARD 0
#    if(__clang_major__ <= 8)
#      pragma clang diagnostic push
#      pragma clang diagnostic ignored "-Wmissing-noreturn"
#      pragma clang diagnostic ignored "-Wunused-parameter"
#      pragma clang diagnostic ignored "-Wdeprecated-declarations"
#    endif
#  endif
# endif
#define __DPX_BOOST_WARNINGS
#else //__DPX_BOOST_WARNINGS
# undef __DPX_BOOST_WARNINGS
# ifdef __GNUC__
#  ifndef __clang__
#    if(__GNUC__ == 7) || (__GNUC__ == 8 && __GNUC_MINOR__ <= 3)
#      pragma GCC diagnostic pop
#    endif
#  endif
#  ifdef __clang__
#    if(__clang_major__ <= 8)
#      pragma clang diagnostic pop
#    endif
#  endif
# endif
#endif
