#include <catch2/catch.hpp>
#include <torch/torch.h>

#include <iostream>

/**!
 * Inspired by https://pytorch.org/tutorials/advanced/cpp_frontend.html
 */
int createNetAndForward(int64_t numInputs, int64_t numNeurons, int64_t numBatch)
{
    struct Net : torch::nn::Module
    {
        Net(int64_t numInputs, int64_t numNeurons)
        {
            W = register_parameter("W", torch::randn({numInputs, numNeurons}));
            b = register_parameter("b", torch::randn(numNeurons));
        }
        
        torch::Tensor forward(torch::Tensor input)
        {
            return torch::addmm(b, input, W);
        }
        
        torch::Tensor W, b;
    };
    
    // Create the net
    Net net(numInputs, numNeurons);
    
    // Pass one random input
    torch::Tensor inputs = torch::rand({numBatch, numInputs});
    torch::Tensor outputs = net.forward(inputs);
    
    return 0;
}

TEST_CASE("Executing a very basic one layer FNN forward pass", "[PyTorch C++ API]")
{
    REQUIRE(createNetAndForward(1, 1, 1) == 0);
    REQUIRE(createNetAndForward(1, 1, 5) == 0);
    REQUIRE(createNetAndForward(1, 3, 1) == 0);
    REQUIRE(createNetAndForward(2, 1, 1) == 0);
    REQUIRE(createNetAndForward(1, 3, 5) == 0);
    REQUIRE(createNetAndForward(2, 3, 1) == 0);
    REQUIRE(createNetAndForward(2, 3, 5) == 0);
}
