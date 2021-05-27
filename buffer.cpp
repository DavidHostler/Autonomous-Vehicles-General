#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>

class Buffer{
    int buffer_capacity = 100000;
    int buffer_counter = 0;
    int batch_size = 64;
    //int obs_tuple[4];
    //The tuple explored
    int state, action, reward, new_state; 
    std::vector<int> state_buffer = {};
    std::vector<int> action_buffer = {};
    std::vector<int> reward_buffer = {};
    std::vector<int> new_state_buffer = {};
    //This assumes that the inputs are scalar (0-dim) floating point numbers;
    //Will have to doublecheck the python code
    void record(float state, float action, float reward, float new_state){
        int index = buffer_counter % buffer_capacity;
        //Buffers
        for(int i =0; i < buffer_capacity; i++){
            if(i == index){
            state_buffer.push_back(state);
            action_buffer.push_back(action);
            reward_buffer.push_back(reward);
            new_state_buffer.push_back(new_state);
            buffer_counter++;
            }
        } 

    }


    void learn(){
        int record_range = std::min(buffer_capacity, buffer_counter);
        int batch_indices; //needs a random generator function likely built into c++

        /*

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


        
        */

    }
    

};



int main(){ 
  
    //Just to check that the code compiled correctly!
    std::cout << "MASTER! MASTER! WHERE'S THE DREAMS THAT I'VE BEEN AFTER!?!" ;
    std::cout << "FIX MEEEEEE!!!" << "Kirk soloing madly";

}
