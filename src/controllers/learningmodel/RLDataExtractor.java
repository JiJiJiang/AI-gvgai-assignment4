/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package controllers.learningmodel;

import tools.*;
import core.game.Observation;
import core.game.StateObservation;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Observable;

import ontology.Types;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author yuy
 */
public class RLDataExtractor {
    public FileWriter filewriter;
    public static Instances s_datasetHeader =datasetHeader();
    
    public RLDataExtractor(String filename) throws Exception{
        
        filewriter = new FileWriter(filename+".arff");
        filewriter.write(s_datasetHeader.toString());
        /*
                // ARFF File header
        filewriter.write("@RELATION AliensData\n");
        // Each row denotes the feature attribute
        // In this demo, the features have four dimensions.
        filewriter.write("@ATTRIBUTE gameScore  NUMERIC\n");
        filewriter.write("@ATTRIBUTE avatarSpeed  NUMERIC\n");
        filewriter.write("@ATTRIBUTE avatarHealthPoints NUMERIC\n");
        filewriter.write("@ATTRIBUTE avatarType NUMERIC\n");
        // objects
        for(int y=0; y<14; y++)
            for(int x=0; x<32; x++)
                filewriter.write("@ATTRIBUTE object_at_position_x=" + x + "_y=" + y + " NUMERIC\n");
        // The last row of the ARFF header stands for the classes
        filewriter.write("@ATTRIBUTE Class {0,1,2}\n");
        // The data will recorded in the following.
        filewriter.write("@Data\n");*/
        
    }

    /**
     * Complete all feature
     * @param features: a piece of data recorder
     * @param action: action index
     * @param reward: Qvalue
     * @return
     */
    public static Instance makeInstance(double[] features, int action, double reward){
        //features[872] = action;
        //features[873] = reward;
        features[s_datasetHeader.numAttributes()-2] = action;
        features[s_datasetHeader.numAttributes()-1] = reward;
        Instance ins = new Instance(1, features);
        ins.setDataset(s_datasetHeader);
        return ins;
    }

    /**
     * Left: 0, Right: 1, Down: 2, Up: 3
     * @param obs : stateObs
     * @param l : all observations
     * @return : nearestDir
     */
    public static int nearestObservationDir(StateObservation obs,ArrayList<Observation> l){
        ArrayList<Observation> grid[][]=obs.getObservationGrid();
        final int []x={-1,1,0,0};
        final int []y={0,0,1,-1};
        Vector2d avatarPos=obs.getAvatarPosition();
        Vector2d[] dir4=new Vector2d[4];
        for(int i=0;i<4;i++){
            dir4[i]=new Vector2d(avatarPos);
            dir4[i].x+=x[i]*20;
            dir4[i].y+=y[i]*20;
        }
        double[] minDist={Double.MAX_VALUE,Double.MAX_VALUE,Double.MAX_VALUE,Double.MAX_VALUE};
        boolean[] isDirValid={true,true,true,true};
        for(int i=0;i<4;i++){
            int xLen=grid.length;
            int yLen=grid[0].length;
            int xIndex=(((int)dir4[i].x)/20+xLen)%xLen;
            int yIndex=(((int)dir4[i].y)/20+yLen)%yLen;
            if(grid[xIndex][yIndex].size()==1
                    &&grid[xIndex][yIndex].get(0).itype==0)
                isDirValid[i]=false;
        }
        /*
        for(int i=0;i<4;i++) {
            System.out.print(isDirValid[i]);
        }
        System.out.println();
        //*/
        for(Observation o:l){
            for(int i=0;i<4;i++){
                if(isDirValid[i]) {
                    minDist[i] = Math.min(minDist[i], o.position.dist(dir4[i]));
                }
            }
        }
        int minDir=0;
        double minDirDist=minDist[0];
        for(int i=1;i<4;i++){
            if(minDist[i]<minDirDist){
                minDir=i;
                minDirDist=minDist[i];
            }
        }

        return minDir;
    }

    /**
     * judge whether avatar is near ghosts.
     * @param obs : stateObs
     * @param allGhost : all ghosts
     * @return yes or no
     */
    public static boolean isNearGhosts(StateObservation obs,ArrayList<Observation> allGhost){
        Vector2d avatarPos=obs.getAvatarPosition();
        double minDist=Double.MAX_VALUE;
        for(Observation o:allGhost) {
            minDist = Math.min(minDist, o.position.dist(avatarPos));
        }
        double threshold=10;
        if(minDist<20*threshold)
            return true;
        else
            return false;
    }

    /**
     * Extract feature from current obs to get a piece of data recorder(without initializing action and Qvalue)
     * @param obs : current stateObservation
     * @return
     */
    public static double[] featureExtract(StateObservation obs){
        //System.out.println(s_datasetHeader.numAttributes());

        // num(pellets+fruit+ghosts+powerpills) + 1 + 1 + 1(action) + 1(Q)
        // 5 + 4(ghosts direct) + 1(action) + 1(Q)
        double[] feature = new double[s_datasetHeader.numAttributes()];
        if( obs.getImmovablePositions()!=null ) {
            for (ArrayList<Observation> l : obs.getImmovablePositions()) {
                if(l.size()==0)
                    continue;
                // pellets
                else if (l.get(0).itype == 4) {
                    feature[0]=nearestObservationDir(obs,l);
                }
                // fruit
                else if (l.get(0).itype == 3) {
                    feature[1]=nearestObservationDir(obs,l);
                }
            }
        }
        // 4 ghosts
        ArrayList<Observation> allGhost = new ArrayList<>();
        if( obs.getPortalsPositions()!=null ) {
            for (ArrayList<Observation> l : obs.getPortalsPositions()){
                allGhost.addAll(l);
            }
        }
        feature[2]=nearestObservationDir(obs,allGhost);
        // isNearGhosts
        feature[5]=isNearGhosts(obs,allGhost)?1:0;
        // 4 powerpills
        if( obs.getResourcesPositions()!=null ) {
            for (ArrayList<Observation> l : obs.getResourcesPositions()){
                feature[3]=nearestObservationDir(obs,l);
            }
        }
        // 4 ghost direct
        Vector2d avatarPos=obs.getAvatarPosition();
        int i=6;
        for(Observation o:allGhost) {
            double delta_x=o.position.x-avatarPos.x;
            double delta_y=o.position.y-avatarPos.y;
            double dist=o.position.dist(avatarPos);
            feature[i] = Math.acos(delta_x/dist)+(delta_y<0?Math.PI:0);
            i++;
        }

        /*
        int[][] map = new int[28][31];
        // Extract features
        LinkedList<Observation> allobj = new LinkedList<>();

        if( obs.getImmovablePositions()!=null ) {
            for (ArrayList<Observation> l : obs.getImmovablePositions()){
                // pellets
                if(l.get(0).itype==4){
                    //System.out.println(l.size());
                }
                // fruit
                if(l.get(0).itype==3){
                    //System.out.println(l.size());
                }
                allobj.addAll(l);
            }
        }
        // 4 ghosts
        if( obs.getPortalsPositions()!=null ) {
            for (ArrayList<Observation> l : obs.getPortalsPositions()){
                //System.out.println(l.get(0).itype);
                allobj.addAll(l);
            }
        }
        // 4 powerpills
        if( obs.getResourcesPositions()!=null ) {
            for (ArrayList<Observation> l : obs.getResourcesPositions()){
                //System.out.println(l.get(0).itype);
                allobj.addAll(l);
            }
        }

        for(Observation o : allobj){
            Vector2d p = o.position;
            int x = (int)(p.x/20); //squre size is 20 for pacman
            int y= (int)(p.y/20);
            map[x][y] = o.itype;
        }
        for(int y=0; y<31; y++)
            for(int x=0; x<28; x++)
                feature[y*28+x] = map[x][y];

        // 4 states
        feature[868] = obs.getGameTick();
        //System.out.println("Tick:"+feature[868]);
        feature[869] = obs.getAvatarSpeed();
        //System.out.println("Speed:"+feature[869]);
        feature[870] = obs.getAvatarHealthPoints();
        //System.out.println("HealthPoint:"+feature[870]);
        feature[871] = obs.getAvatarType();
        //System.out.println("Avatar type:"+feature[871]);

        */
        return feature;
    }

    /**
     * Make training dataSet header.
     * @return dataSetHeader(without data)
     */
    public static Instances datasetHeader(){

        if (s_datasetHeader!=null)
            return s_datasetHeader;

        /*
        // 28*31 + 4 + 1 + 1 locations
        FastVector attInfo = new FastVector();
        // 28*31
        for(int y=0; y<28; y++){
            for(int x=0; x<31; x++){
                Attribute att = new Attribute("object_at_position_x=" + x + "_y=" + y);
                attInfo.addElement(att);
            }
        }
        */
        /*
        FastVector attInfo = new FastVector();
        Attribute att;
        if( initStateObs.getImmovablePositions()!=null ) {
            for (ArrayList<Observation> l : initStateObs.getImmovablePositions()){
                // pellets
                if(l.get(0).itype==4){
                    for(Observation o:l) {
                        int x = (int) (o.position.x / 20);
                        int y = (int) (o.position.y / 20);
                        att = new Attribute("pellet_at_position_x=" + x + "_y=" + y);
                        attInfo.addElement(att);
                    }
                }
                // fruit
                else if(l.get(0).itype==3){
                    for(Observation o:l) {
                        int x = (int) (o.position.x / 20);
                        int y = (int) (o.position.y / 20);
                        att = new Attribute("fruit_at_position_x=" + x + "_y=" + y);
                        attInfo.addElement(att);
                    }
                }
            }
        }
        // 4 ghosts
        if( initStateObs.getPortalsPositions()!=null ) {
            for (ArrayList<Observation> l : initStateObs.getPortalsPositions()){
                int x = (int) (l.get(0).position.x / 20);
                int y = (int) (l.get(0).position.y / 20);
                att = new Attribute("ghost_at_position_x=" + x + "_y=" + y);
                attInfo.addElement(att);
            }
        }
        // 4 powerpills
        if( initStateObs.getResourcesPositions()!=null ) {
            for (ArrayList<Observation> l : initStateObs.getResourcesPositions()){
                for(Observation o:l) {
                    int x = (int) (o.position.x / 20);
                    int y = (int) (o.position.y / 20);
                    att = new Attribute("powerpill_at_position_x=" + x + "_y=" + y);
                    attInfo.addElement(att);
                }
            }
        }
        */
        FastVector attInfo = new FastVector();
        Attribute att;
        // nearestPelletDir
        FastVector nearestPelletDir = new FastVector();
        nearestPelletDir.addElement("0");
        nearestPelletDir.addElement("1");
        nearestPelletDir.addElement("2");
        nearestPelletDir.addElement("3");
        att = new Attribute("nearestPelletDir", nearestPelletDir);
        attInfo.addElement(att);
        // nearestFruitDir
        FastVector nearestFruitDir = new FastVector();
        nearestFruitDir.addElement("0");
        nearestFruitDir.addElement("1");
        nearestFruitDir.addElement("2");
        nearestFruitDir.addElement("3");
        att = new Attribute("nearestFruitDir", nearestFruitDir);
        attInfo.addElement(att);
        // nearestGhostDir
        FastVector nearestGhostDir = new FastVector();
        nearestGhostDir.addElement("0");
        nearestGhostDir.addElement("1");
        nearestGhostDir.addElement("2");
        nearestGhostDir.addElement("3");
        att = new Attribute("nearestGhostDir", nearestGhostDir);
        attInfo.addElement(att);
        // nearestPowerpillDir
        FastVector nearestPowerpillDir = new FastVector();
        nearestPowerpillDir.addElement("0");
        nearestPowerpillDir.addElement("1");
        nearestPowerpillDir.addElement("2");
        nearestPowerpillDir.addElement("3");
        att = new Attribute("nearestPowerpillDir", nearestPowerpillDir);
        attInfo.addElement(att);
        // isNearGhosts
        FastVector isNearGhosts = new FastVector();
        isNearGhosts.addElement("0");
        isNearGhosts.addElement("1");
        att = new Attribute("isNearGhosts", isNearGhosts);
        attInfo.addElement(att);
        // 4 Ghosts direct
        for (int i=1;i<=4;i++) {
            att = new Attribute("ghost"+i+"direct");
            attInfo.addElement(att);
        }
        // GameTick
        //att = new Attribute("GameTick" ); attInfo.addElement(att);
        //action
        FastVector actions = new FastVector();
        actions.addElement("0");
        actions.addElement("1");
        actions.addElement("2");
        actions.addElement("3");
        att = new Attribute("actions", actions);        
        attInfo.addElement(att);
        // Q value
        att = new Attribute("Qvalue");
        attInfo.addElement(att);
        
        Instances instances = new Instances("PacmanQdata", attInfo, 0);
        instances.setClassIndex( instances.numAttributes() - 1);
        
        return instances;
    }
    
}
