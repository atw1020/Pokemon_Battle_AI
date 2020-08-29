//
//  main.cpp
//  Bytes to bits converter
//
//  Created by arthur wesley on 8/5/19.
//  Copyright © 2019 arthur wesley. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <Python.h>
#include <string>
#include <filesystem>


const char* version = "0.2.0";

// file delimiters
const char* team_delimiter = "td"; // team data
const char* turns_delimiter = "tr"; // turn
const char* winner_delimiter = "wd"; // winner data

// Team Preview Delimiters
const char* first_player_team_delimiter = "ft"; // first team
const char* second_player_team_delimiter = "st"; // second team

// turn delimiters
const char* new_turn_delimiter = "nt"; // new turn

const char* first_player_switch_delimiter = "fs"; // first switch
const char* first_player_attacker_delimiter = "fa"; // first attacker
const char* first_player_attack_delimiter = "fm"; // first move
const char* first_player_cant_delimiter = "fc"; // first can't
const char* first_player_ID_delimiter = "fi"; // first ID

const char* second_player_switch_delimiter = "ss"; // second switch
const char* second_player_attacker_delimiter = "sa"; // second attacker
const char* second_player_attack_delimiter = "sm"; // second move
const char* second_player_cant_delimiter = "sc"; // second can't
const char* second_player_ID_delimiter = "si"; // second ID

const char* other_attributes_delimiter = "oa"; // other attributes

using namespace std;

vector<vector<int> > mk_lookup_table(){
    /*
     
     makes a lookup table
     
     */
    
    vector<vector<int> > table(256);
    vector<int> subtable(8);
    
    for(int i = 0; i < 256; i++){
        table[i] = subtable;
        
        for(int j = 0; j < 8; j++){
            table[i][j] = (i >> (7 - j)) & 1;
        }
    }
    
    return table;
}

PyObject *Cpp_Vector_to_Python_List(vector<int> input){
    
    // first allocate the memory by creating a python list the same size as the input vector
    
    long int len = input.size();
    PyObject* output = PyList_New(len);
    
    return output;
}

void print_binary(string bytes){
    
    short int size;
    vector<vector<int> > table;
    table = mk_lookup_table();
    
    // get the length of the string and use it to create the bit array
    size = bytes.length();
    int data[8 * size];
    
    /*
     cout << bytes << endl;
     */
    
    // now go through the bytes and find their corresponding string
    for(int i = 0; i < size; i++){
        /*
         cout << bytes[i] << ": ";
         */
        
        for(int j = 0; j < 8; j++){
            data[8 * i + j] = table[(unsigned char) bytes[i]][j];
            
            
            /*
             cout << data[8 * i + j];
             
             // if we aren't at the end of the line, print a comma
             
             if( j != 7){
             cout << ", ";
             }
             */
            
        }
        /*
         cout << endl;
         */
    }
    
}

int num_instances(string sub, string main){
    /*
     
     finds the number of instances of substrin sub in string main
     
     */
    
    int i = 0;
    short int searched = main.find(sub);
    
    while(searched == -1){
        searched = main.find(sub, searched + 1);
        i++;
    }
    
    return i;
}

string get_bytes(string path){
    
    string bytes;
    
    // open the file
    ifstream replay_file(path, ios::in | ios::binary);
    
    if(not replay_file.is_open()){
        // if the file isn't open tell us and exit the program
        std::cout << "error opening file" << endl;
        return NULL;
    }
    
    // we can now assign the bytes using this fancy code I got off the internet
    bytes.assign((istreambuf_iterator<char>(replay_file)),
                  istreambuf_iterator<char>()           );

    replay_file.close();
    
    return bytes;
}

void convert_binary_no_lookup_table(string Bytes, int data[]){
    /*
     
     converts a binary file to an array of integers without using a lookup table
     
     */
    
    short int size = Bytes.length();
    
    for(int i = 0; i < size; i++){
        for(int j = 0; j < 8; j++){
            data[8 * i + j] = (Bytes[i] >> (7 - j)) & 1;
            //cout << data[8 * i + j];
        }
    }
    
}

long int get_size(string Bytes){
    /*
     
     takes in a 4 byte number as a string and converts it to a long int
     
     */
    
    long int size;
    
    size = (((int) (unsigned char) Bytes[0]) << 24) + (((int) (unsigned char) Bytes[1]) << 16) + (((int) (unsigned char) Bytes[2]) << 8) + ((int) (unsigned char) Bytes[3]);
    
    return size;
    
}

long int count_turns(string &Bytes){
    /*
     
     counts the number of turns in a replay file
     
     */
    
    // find the number of instances of the first player switch delimiter in Bytes
    
    int file_size = Bytes.length();
    int delimiter_size = 2; // change this to be the length of the first player switch delimiter
    int j;
    
    int num_instances = 0;
    
    for(int i = 0; i < file_size - delimiter_size; i++){
        // check to see if the characters match
        j = 0;
        
        while(Bytes[i + j] == first_player_switch_delimiter[j]){
            
            j++;
            if(j == delimiter_size){
                // add one to the number of instances of the turn we found
                num_instances++;
            }
        }
    }
    
    return num_instances;
    
}

PyObject* bytes_to_py_list(string Bytes, const string delimiter, long int &starting_index){
    /*
     
     takes in a string of bytes and returns a python list where each item in the list corresponds to one bit
     of the bytes in the Bytes string
     
     the delimiter indicates the start of the section of bytes we are looking for and the starting index tells us where to start looking for that delimiter
     
     */
    
    // first thing we need to do is find the delimiter
    
    long int index, j;
    
    //cout << starting_index << endl;
    
    index = Bytes.find(delimiter, starting_index) + delimiter.length();
    // update the starting index
    starting_index = index;
    
    // now slice off all the bytes before the index we are intrested in
    Bytes = Bytes.substr(index);
    
    // now that we've found the data, we know that the next four bytes tell us the size of this section
    long int size = get_size(Bytes.substr(0, 4)); // note that this size tells us the number of *bits* in our *output list* and not the number of
                                                  // *bytes* in our input string
    
    // and slize the first four Bytes off of the Bytes string
    Bytes = Bytes.substr(4);
    
    // declare the output pylist for speed
    PyObject* output = PyList_New(size);
    PyObject* next_item = NULL;
    
    for(long int i = 0; i < (size / 8) + 1; i++){
        
        for(j = 0; j < 8; j++){

            if(8 * i + j < size){
                
                // if we are in inside the range of the list write the next bit
                
                // set the value of the next item to be put into the output list
                next_item = PyLong_FromLong((Bytes[i] >> j) & 1);
                //Py_INCREF(next_item);

                if(PyList_SetItem(output, 8 * i + j, next_item) != 0){
                    // return an error
                    return NULL;
                }

                //if(next_item != Py_None){
                //    Py_DECREF(next_item);
                //}
            }
        }
    }

    /*
    // finally since we are re-using the next item we need to delete it to free memory
    if(next_item != Py_None){
        Py_XDECREF(next_item);
    }
     //*/
    
    return output;
}

PyObject* _read_replay_file(string path){
    /*
     
     reads in a file located at path
     
     */
    
    // declare variables
    string Bytes;
    long int starting_index = 0;
    long int turns;
    
    // start by getting the bytes from the file
    Bytes = get_bytes(path);
    
    // count the number of turns in the replay
    turns = count_turns(Bytes);
    
    
    
    
    
    // declare the Python List
    PyObject *output = PyList_New(3);
    
    if(PyList_SetItem(output, 0, PyList_New(2)) == -1){ // the first list contains the team data
        // if this function fails return an error
        return NULL;
    }
    if(PyList_SetItem(output, 1, PyList_New(turns)) == -1){ // the second list contains the turns
        // return an error to python
        return NULL;
    }
    if(PyList_SetItem(output, 2, PyList_New(1)) == -1){ // the third list contains the winner
        // return an error to python
        return NULL;
    }
    
    
    
    
    
    
    // oh boy time for some ugly function calls
    
    // start out by setting the team data
    if(PyList_SetItem(PyList_GetItem(output, 0), 0, bytes_to_py_list(Bytes, first_player_team_delimiter, starting_index)) == -1){
        // return an error to python
        return NULL;
    }
    
    if(PyList_SetItem(PyList_GetItem(output, 0), 1, bytes_to_py_list(Bytes, second_player_team_delimiter, starting_index)) == -1){
        // return an error to python
        return NULL;
    }
    
    
    // next loop through the turns and get the data out of them
    
    for(int i = 0; i < turns; i++){
        
        // start by setting this item of the team list to an empty pylist
        if(PyList_SetItem(PyList_GetItem(output, 1), i, PyList_New(11)) == -1){
            // return an error to python
            return NULL;
        }
        
        
        // now set the items of this turn based on their delimiter
        
        // first the first player's items
        
        if(PyList_SetItem(PyList_GetItem(PyList_GetItem(output, 1), i), 0, bytes_to_py_list(Bytes, first_player_switch_delimiter, starting_index)) == -1){
            // return an error to python
            return NULL;
        }
        
        if(PyList_SetItem(PyList_GetItem(PyList_GetItem(output, 1), i), 1, bytes_to_py_list(Bytes, first_player_attacker_delimiter, starting_index)) == -1){
            // return an error to python
            return NULL;
        }
        
        if(PyList_SetItem(PyList_GetItem(PyList_GetItem(output, 1), i), 2, bytes_to_py_list(Bytes, first_player_attack_delimiter, starting_index)) == -1){
            // return an error to python
            return NULL;
        }
        
        if(PyList_SetItem(PyList_GetItem(PyList_GetItem(output, 1), i), 3, bytes_to_py_list(Bytes, first_player_cant_delimiter, starting_index)) == -1){
            // return an error to python
            return NULL;
        }
        
        if(PyList_SetItem(PyList_GetItem(PyList_GetItem(output, 1), i), 4, bytes_to_py_list(Bytes, first_player_ID_delimiter, starting_index)) == -1){
            // return an error to python
            return NULL;
        }
        
        
        // next the second player's items
        
        if(PyList_SetItem(PyList_GetItem(PyList_GetItem(output, 1), i), 5, bytes_to_py_list(Bytes, second_player_switch_delimiter, starting_index)) == -1){
            // return an error to python
            return NULL;
        }
        
        if(PyList_SetItem(PyList_GetItem(PyList_GetItem(output, 1), i), 6, bytes_to_py_list(Bytes, second_player_attacker_delimiter, starting_index)) == -1){
            // return an error to python
            return NULL;
        }
        
        if(PyList_SetItem(PyList_GetItem(PyList_GetItem(output, 1), i), 7, bytes_to_py_list(Bytes, second_player_attack_delimiter, starting_index)) == -1){
            // return an error to python
            return NULL;
        }
        
        if(PyList_SetItem(PyList_GetItem(PyList_GetItem(output, 1), i), 8, bytes_to_py_list(Bytes, second_player_cant_delimiter, starting_index)) == -1){
            // return an error to python
            return NULL;
        }
        
        if(PyList_SetItem(PyList_GetItem(PyList_GetItem(output, 1), i), 9, bytes_to_py_list(Bytes, second_player_ID_delimiter, starting_index)) == -1){
            // return an error to python
            return NULL;
        }
        
        // finally the other attributes
        
        if(PyList_SetItem(PyList_GetItem(PyList_GetItem(output, 1), i), 10, bytes_to_py_list(Bytes, other_attributes_delimiter, starting_index)) == -1){
            // return an error to python
            return NULL;
        }
        
    }
    
    // and finally set the winner
    
    if(PyList_SetItem(output, 2, bytes_to_py_list(Bytes, winner_delimiter, starting_index)) == -1){
        // return an error to python
        return NULL;
    }
    
    return output;
    
}

bool Write_PyList_File(ofstream &output_file, string Delimiter, PyObject *List){
    /*
     
     writes a python list (which MUST contain integers where only ONE bit is significant) to an output file
     
     */
    
    // the first thing we need to do is check to see if all the items in this list are of PyLong Type
    
    long int len = PyList_Size(List);
    PyObject *List_Item;
    
    for(long int i = 0; i < len; i++){
        List_Item = PyList_GetItem(List, i);
        if(not PyLong_Check(List_Item)){
            return false;
        }
        // to reuse the list item delete it at the end of each iteration
        // Py_DECREF(List_Item);
    }

    // now that we know that our list contains longs we can declare some more variables
    
    PyObject *SubList;
    char output_byte;
    int j;
    
    // start by writing the Delimiter and size of the list

    // write the delimiter
    output_file << Delimiter;

    // next, write 4 chars for the len
    
    output_file << (char) ((len >> 24) & 255) << (char) ((len >> 16) & 255) << (char) ((len >> 8) & 255) << (char) (len & 255);
    
    // now step through the list in increments of 8 items
    
    for(long int i = 0; i < (len / 8) + 1; i++){
        
        // slice out the sublist
        SubList = PyList_GetSlice(List, 8 * i, 8 * (i + 1));
        
        // compute the output byte
        output_byte = 0;
        
        // go through the sublist and write the items to the output file
        for(j = 0; j < 8; j++){
            
            if(8 * (i + 1) - j <= len){
                // add the new bit to the output byte using a bitwise "or" operation
                // and bit shifting the rest of the byte
                
                output_byte = (output_byte << 1) | PyLong_AsLong(PyList_GetItem(SubList, 7 - j));
            }
            else{
                // if we go out of range of the list shift the bytes the remaining distance
                output_byte = output_byte << 1;
            }
        }
        
        // with the output byte constructed we can write the byte to the output file
        output_file << output_byte;
        
        // since we are re-using the sublist object we need to delete the sublist at the end of the loop
        // in order to delete a python object we have to tell the python garbage collector what we are doing
        // so we need to use the Py_DECREF() function to delete the SubList
        if(SubList != Py_None){
            Py_DECREF(SubList);
        }
        
    }
    
    return true;
    
}

PyObject* _mk_output_file(PyObject *Input_List, string filepath){
    /*
     
     takes a python list and writes it to a file with the output file path
     
     */
    
    
    
    // first let's unpack the Input List into it's 3 components
    //     - The Team Preview Data
    //     - The Data about the Turns
    //     - The Winner Information
    
    PyObject *Player_1_Team, *Player_2_Team;
    PyObject *Turns_Data;
    PyObject *Winner_Data;
    
    
    // now let's open the output file
    ofstream output_file(filepath, ios::out | ios::binary);



    
    // first thing we do is get the player1 and player2 lists
    
    Player_1_Team = PyList_GetItem(PyList_GetItem(Input_List, 0), 0);
    Player_2_Team = PyList_GetItem(PyList_GetItem(Input_List, 0), 1);
    
    Turns_Data = PyList_GetItem(Input_List, 1);
    Winner_Data = PyList_GetItem(Input_List, 2);
    
    // now write the data to the file
    
    if(not output_file.is_open()){
        cout << "could not open specified flie" << endl;
        return NULL; // tell python that an error occured
    }
    
    
    
    // write the team data

    //output_file << team_delimiter;
    
    // start by writing the Team data Delimiter
    
    if(not Write_PyList_File(output_file, first_player_team_delimiter, Player_1_Team)){
        // raise an error in python
        return NULL;
    }
    
    if(not Write_PyList_File(output_file, second_player_team_delimiter, Player_2_Team)){
        // rasie an error in python
        return NULL;
    }
    
    
    
    
    // write the turns data

    long num_turns = PyList_Size(Turns_Data);
    
    //output_file << turns_delimiter;

    // go through all the turns
    
    for(int i = 0; i < num_turns; i++){
        // each new turn write the new turn delimiter
        
        //output_file << new_turn_delimiter;
        
        // now write all the other details about the turn
        // write all the data about the first player
        
        // first player switch
        if(not Write_PyList_File(output_file, first_player_switch_delimiter, PyList_GetItem(PyList_GetItem(Turns_Data, i), 0))){
            // raise an error in python
            return NULL;
        }
        
        // first player attacker
        if(not Write_PyList_File(output_file, first_player_attacker_delimiter, PyList_GetItem(PyList_GetItem(Turns_Data, i), 1))){
            // raise an error in python
            return NULL;
        }
        
        // first player attack
        if(not Write_PyList_File(output_file, first_player_attack_delimiter, PyList_GetItem(PyList_GetItem(Turns_Data, i), 2))){
            // raise python error
            return NULL;
        }
        
        // first player can't attack
        if(not Write_PyList_File(output_file, first_player_cant_delimiter, PyList_GetItem(PyList_GetItem(Turns_Data, i), 3))){
            // raise python error
            return NULL;
        }
        
        // first player ID
        if(not Write_PyList_File(output_file, first_player_ID_delimiter, PyList_GetItem(PyList_GetItem(Turns_Data, i), 4))){
            // raise python error
            return NULL;
        }




        // write second player data
        
        // second player switch
        if(not Write_PyList_File(output_file, second_player_switch_delimiter, PyList_GetItem(PyList_GetItem(Turns_Data, i), 5))){
            // raise an error in python
            return NULL;
        }
        
        // second player attacker
        if(not Write_PyList_File(output_file, second_player_attacker_delimiter, PyList_GetItem(PyList_GetItem(Turns_Data, i), 6))){
            // raise an error in python
            return NULL;
        }
        
        // second player attack
        if(not Write_PyList_File(output_file, second_player_attack_delimiter, PyList_GetItem(PyList_GetItem(Turns_Data, i), 7))){
            // raise python error
            return NULL;
        }
        
        // second player can't attack
        if(not Write_PyList_File(output_file, second_player_cant_delimiter, PyList_GetItem(PyList_GetItem(Turns_Data, i), 8))){
            // raise python error
            return NULL;
        }
        
        // second player ID
        if(not Write_PyList_File(output_file, second_player_ID_delimiter, PyList_GetItem(PyList_GetItem(Turns_Data, i), 9))){
            // raise python error
            return NULL;
        }
        
        // now write the other attributes section
        
        if(not Write_PyList_File(output_file, other_attributes_delimiter, PyList_GetItem(PyList_GetItem(Turns_Data, i), 10))){
            // raise an error in python
            return NULL;
        }
        
        //return Py_None;
        
        
    }
    
    // write the winner data
    
    if(not Write_PyList_File(output_file, winner_delimiter, Winner_Data)){
        // raise an error in python
        return NULL;
    }
    
    // close the file
    output_file.close();
    
    return Py_None;
}


// python interfacing

PyObject* mk_output_file(PyObject *self, PyObject *args){
    /*
     
     function that is called in python
     saves data about a replay into an output file
     
     */
    const char *filepath;
    PyObject *data;
    
    if(not PyArg_ParseTuple(args, "sO", &filepath, &data)){
        // if we don't have the right arguments raise a python error
        return NULL;
    }
    
    // run the function
    
    _mk_output_file(data, filepath);
    
    return Py_None;
    
}

PyObject* read_replay_file(PyObject *self, PyObject *args){
    /*
     
     function that is called in python
     loads data about a replay into python
     
     */
    
    const char *filepath;
    
    if(not PyArg_ParseTuple(args, "s", &filepath)){
        return NULL;
    }
    
    return _read_replay_file(filepath);
    
}

PyObject *get_version(PyObject *self, PyObject *args){
    // give python the version data
    std::cout << version << endl;
    return Py_None;
}

// make a list of all the methods that will be passed into python
static PyMethodDef Methods[] = {
    {"make_replay_file", mk_output_file, METH_VARARGS, "creates a replay file from a python list"},
    {"read_replay_file", read_replay_file, METH_VARARGS, "reads a replay file into a python list"},
    {"version", get_version, METH_NOARGS, "returns the version of this software"},
    {NULL, NULL, 0, NULL}
};

// make the module
// note: this will only compile for Python 3
static struct PyModuleDef pokereplay = {
    PyModuleDef_HEAD_INIT,
    "pokereplay",
    "reads and writes pokemon showdown battle replays",
    -1,
    Methods
};
PyMODINIT_FUNC PyInit_pokereplay(void) {
    Py_Initialize();
    return PyModule_Create(&pokereplay);
}
