import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;

public class Shavadoop_Master {
	
	private String l_filepath; 
	private String l_FileName;
	private String l_listemachine;
	private int  l_N_Slave;
	private ArrayList<String> ListeMachineUsable;
	private HashMap<Integer,String> MapperList;
	private HashMap<String,ArrayList<Integer>> UMdico;
	private HashMap<String,String> ReducerList;
	
	public Shavadoop_Master (String Filepath, String FileName, int N_Slave, String SlaveMachine){  
		l_filepath = Filepath;
		l_FileName = FileName;
		l_N_Slave = N_Slave;
		l_listemachine = SlaveMachine;
		ListeMachineUsable = new ArrayList<String>();
		MapperList = new HashMap<Integer,String>();
		UMdico = new HashMap<String,ArrayList<Integer>>();
		ReducerList = new HashMap<String,String>();
	}
	
	 public String cleanText(String line) {
          String clean = line;
          // clean the non alpha numeric character or space
          clean = clean.replaceAll("[^a-zA-Z0-9\\s]", "");
          // just one space beetween each words
          clean = clean.replaceAll("\\s+", " ");
          return clean;
  }
	 
	/* Execution du split fichier */
	 
	public void Splitfile() throws IOException{
		
		System.out.println("----------------------------------------------------");
		System.out.println("File Spliting begins");
		
		int startTime = (int) System.currentTimeMillis();
		/* recupere le fichier au format file pour faire le split */
		File dir_S = new File (l_filepath + "/Split");
		FileUtils.deleteDirectory(dir_S);
		FileUtils.forceMkdir(dir_S);
		File fichier = new File(l_filepath +"/"+ l_FileName);
		List<String> Lines = FileUtils.readLines(fichier, "UTF-8");
		ArrayList<File> Splits = new ArrayList<File>();
		
		for (int i=0; i < l_N_Slave;i++){
			File split = new File(l_filepath +"/Split/"+ "S_" + new Integer(i).toString());
			FileUtils.deleteQuietly(split);
			split.createNewFile();
			Splits.add(split);
		}
		for (int j = 0;j<l_N_Slave;j++){ 
			String ecriture = StringUtils.join(Lines.subList((j*Lines.size())/l_N_Slave , ((j+1)*Lines.size())/l_N_Slave),"\n");
			FileUtils.write(Splits.get(j), ecriture);
		}
		int endTime = (int) System.currentTimeMillis();
		System.out.println("Temps du split = " + new Integer(endTime - startTime).toString() + " ms" );
		System.out.println("File Spliting ends");
	}
	
	
	/* Recupere les machines a partir d'une liste donnee en argument */
	
	public void getMachine() throws IOException, InterruptedException{
		 /* Lecture du Fichier */
	    String ligne = null;
		BufferedReader bufferFichierMachines = new BufferedReader(new FileReader(l_filepath + "/" + l_listemachine));
	    ArrayList<String> ListeMachine = new ArrayList<String>();
	    while ((ligne = bufferFichierMachines.readLine()) != null){
	    	ListeMachine.add(ligne);
	    }
	    bufferFichierMachines.close();
	    HashMap<Integer,ArrayList<String>> HostCmd = new HashMap<Integer,ArrayList<String>>() ;
	    Integer i = new Integer(0);
	    for(String host: ListeMachine){
	    	ArrayList<String> CMD = new ArrayList<String>();
	    	CMD.add(host);
	    	CMD.add("sleep " + (int) 0);
	    	HostCmd.put(i,CMD);
	    	i+=1;
	      		}
	   	shepherd sleep = new shepherd(HostCmd);
	   	sleep.exec();
	   	ListeMachineUsable = sleep.Usable();
	}
	
	/* Depart des jobs Mapper en distribué */
	
	public void Mapper () throws InterruptedException, IOException{
		System.out.println("----------------------------------------------------");
		System.out.println("Mapper Jobs begin");
		int startTime = (int) System.currentTimeMillis();
		
		HashMap<Integer,ArrayList<String>> HostCmd = new HashMap<Integer,ArrayList<String>>() ;
		File dir_UM = new File (l_filepath + "/UnsortedMap");
		FileUtils.deleteDirectory(dir_UM);
		FileUtils.forceMkdir(dir_UM);
		
		File dir_keys = new File (l_filepath + "/Keys");
		FileUtils.deleteDirectory(dir_keys);
		FileUtils.forceMkdir(dir_keys);
		
		for (int i=0; i< l_N_Slave; i++){
			ArrayList<String> CMD = new ArrayList<String>();
	    	CMD.add(ListeMachineUsable.get(i));
	    	CMD.add("java -jar /cal/homes/johayon/INF727/Mapper.jar" +" "+ new Integer(i).toString() +" " + new Integer(l_N_Slave).toString() + " " +  "/cal/homes/johayon/INF727");
			HostCmd.put(i,CMD);
			MapperList.put(new Integer(i),ListeMachineUsable.get(i));
		}
		shepherd map = new shepherd(HostCmd);
		map.exec();
		int endTime = (int) System.currentTimeMillis();
		System.out.println("Temps des Mappers = " + new Integer(endTime - startTime).toString() + " ms" );
		System.out.println("Mapper Jobs end");
	}
	
	/* Liste des machines utilisées pour le mapper */
	
	public HashMap<Integer,String> getMasterMapper (){
		return MapperList;
	}
	
	/* Liste UMlocation */
	
	public void UMlocation () throws IOException{
		for (int i=0;i<l_N_Slave;i++){
			File keysFile = new File(l_filepath +"/Keys/" +  "keys_" + new Integer(i).toString());
			List<String> Lines = FileUtils.readLines(keysFile, "UTF-8");
			for(String key : Lines){
				if (UMdico.containsKey(key) == false ){
					ArrayList<Integer> tmp = new ArrayList<Integer>();
					tmp.add(new Integer(i));
					UMdico.put(key,tmp);
				}
				else{
					UMdico.get(key).add(new Integer(i));
				}
			} 
		}
	}
		
	public void dico (){
		System.out.println(UMdico);
	}
	
	/* Depart des jobs Shuffler en distribué */
	
	public void shuffle() throws InterruptedException, IOException {
		System.out.println("----------------------------------------------------");
		System.out.println("Shuffle Jobs begin");
		int startTime = (int) System.currentTimeMillis();
		HashMap<Integer,ArrayList<String>> HostCmd = new HashMap<Integer,ArrayList<String>>() ;
		
		File dir_SM = new File (l_filepath + "/SortedMap");
		FileUtils.deleteDirectory(dir_SM);
		FileUtils.forceMkdir(dir_SM);
		for(int j=0; j < l_N_Slave;j++){
			ArrayList<String> CMD = new ArrayList<String>();
	    	CMD.add(ListeMachineUsable.get(j));
	    	CMD.add("java -jar /cal/homes/johayon/INF727/Shuffler.jar "+  new Integer(l_N_Slave).toString() +  " " + new Integer(j).toString() +  " /cal/homes/johayon/INF727");
			HostCmd.put(new Integer(j),CMD);
		}
		shepherd shuf = new shepherd(HostCmd);
		shuf.exec();
		int endTime = (int) System.currentTimeMillis();
		System.out.println("Temps des Shufflers = " + new Integer(endTime - startTime).toString() + " ms" );
		System.out.println("Shuffle Jobs end");
	}
	
	/* Depart des jobs Reducer en distribué */
	
	public void reducer() throws InterruptedException, IOException {
		System.out.println("----------------------------------------------------");
		System.out.println("Reducer Jobs begin");
		int startTime = (int) System.currentTimeMillis();
		HashMap<Integer,ArrayList<String>> HostCmd = new HashMap<Integer,ArrayList<String>>() ;
		File dir_RM = new File (l_filepath + "/ReducedMap");
		FileUtils.deleteDirectory(dir_RM);
		FileUtils.forceMkdir(dir_RM);
		for(int i=0; i<l_N_Slave;i++){
			ArrayList<String> CMD = new ArrayList<String>();
	    	CMD.add(ListeMachineUsable.get(i %l_N_Slave));
	    	CMD.add("java -jar /cal/homes/johayon/INF727/Reducer.jar" +" "+ new Integer(i).toString() +" "+  "/cal/homes/johayon/INF727");
			HostCmd.put(new Integer(i),CMD);
			ReducerList.put(new Integer(i).toString(),ListeMachineUsable.get(i %l_N_Slave));
		}
		shepherd red = new shepherd(HostCmd);
		red.exec();	
		int endTime = (int) System.currentTimeMillis();
		System.out.println("Temps des Reducers = " + new Integer(endTime - startTime).toString() + " ms" );
		System.out.println("Reducer Jobs end");
	}
	
	/* Liste des machines utilisées pour le reducer */
	
	public HashMap<String,String> getWorkerReducer (){
		return ReducerList;
	}
	
	/* Depart ecriture resultat*/
	
	public void results() throws IOException{
		System.out.println("----------------------------------------------------");
		System.out.println("Results writing begin");
		int startTime = (int) System.currentTimeMillis();
		File res = new File(l_filepath +"/"+ "results");
		FileUtils.deleteQuietly(res);
		res.createNewFile();
		
		for(String key : ReducerList.keySet()){
			File fichier = new File(l_filepath + "/ReducedMap/"+"red_" + key);
			List<String> Lines = FileUtils.readLines(fichier, "UTF-8");
			String ecriture = StringUtils.join(Lines,"\n");
			FileUtils.write(res, ecriture,true);
			
		}
		int endTime = (int) System.currentTimeMillis();
		System.out.println("Temps ecriture final = " + new Integer(endTime - startTime).toString() + " ms" );
		System.out.println("Results writing end");
	}
	
	
	
}