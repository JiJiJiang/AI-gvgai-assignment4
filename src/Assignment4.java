import java.lang.annotation.Repeatable;
import java.util.Random;
import core.competition.CompetitionParameters;

import core.ArcadeMachine;
import ontology.effects.unary.Spawn;

/**
 * Created with IntelliJ IDEA.
 * User: Diego
 * Date: 04/10/13
 * Time: 16:29
 * This is a Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class Assignment4
{
    public static void main(String[] args)
    {
        //Reinforcement learning controllers:
    	String rlController = "controllers.learningmodel.Agent";
        
        //Only pacman game
        String game = "examples/gridphysics/pacman.txt";
        String level = "examples/gridphysics/pacman_lvl";

        //Other settings
        boolean visuals = true;
        int seed = new Random().nextInt();

        //Game and level to play
        
        // Monte-Carlo RL training
        CompetitionParameters.ACTION_TIME = 1000000;
        //ArcadeMachine.runOneGame(game, level, visuals, rlController, null, seed, false);
        //String level2 = gamesPath + games[gameIdx] + "_lvl" + 1 +".txt";
        for(int j=0;j<4;j++) {
            for (int i = 0; i < 5; i++) {
                String levelfile = level + i + ".txt";
                long start = System.currentTimeMillis();
                ArcadeMachine.runOneGame(game, levelfile, visuals, rlController, null, seed, false);
                long end = System.currentTimeMillis();
                System.out.println("Run time: " + ((end - start) / 1000) + " s");
            }
            System.out.println();
            System.out.println();
        }
    }
}
