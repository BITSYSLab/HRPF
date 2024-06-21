/*
 * @Author: your name
 * @Date: 2022-02-07 21:38:38
 * @LastEditTime: 2022-02-24 11:05:04
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_NEW_VERSION\HRPA_NEW\framework\config.h
 */
#pragma once

#ifndef ASNC
#define ASNC  1
#endif

#ifndef STREAM_NUM_
#define STREAM_NUM_  4
#endif

#ifndef HANDLE_NUM_
#define HANDLE_NUM_  0
#endif

#ifndef c_num
#define c_num  2
#endif

#ifndef g_num
#define g_num  1
#endif

// #ifndef PARALLEL_FOR
//仅循环需要置为1
#define PARALLEL_FOR 1
// #endif

#define _REDUCE 0
#define T_SIZE (( c_num ) + ( g_num ))