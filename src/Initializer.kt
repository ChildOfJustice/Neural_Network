import versions.Version3

fun main(){
    //There you can run the chosen version:
    val NN_V3 = Version3()
    NN_V3.loadWeightsFromFile = true
    NN_V3.main()
}