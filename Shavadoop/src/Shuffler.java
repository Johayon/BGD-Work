import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;

public class Shuffler {

	public static void main(String[] args) throws IOException, InterruptedException{
		
		/* Reucperation des arguments */
		String l_filepath = args[2];
		int number = Integer.parseInt(args[1]);
		int l_N_Slave = Integer.parseInt(args[0]);
		System.out.println("Thread Shuffler " + new Integer(number).toString() + " begins" );
		HashMap<String,ArrayList<String>> livestock = new HashMap<String,ArrayList<String>>();
		
		/* Lecture des fichiers UMs */
		for(int i=0;i<l_N_Slave;i++){ 
			File fichier = new File(l_filepath +"/UnsortedMap/"+ "UM_" + i + "R_" + number);
			List<String> Lines = FileUtils.readLines(fichier, "UTF-8");
			for(String line : Lines){
				String[] KeyValue = line.split(" ");
				if(livestock.containsKey(KeyValue[0])){
					 livestock.get(KeyValue[0]).add(KeyValue[1]);
				}
				else{
					ArrayList<String> tmp = new ArrayList<String>();
					tmp.add(KeyValue[1]);
					livestock.put(KeyValue[0],tmp);
				}
			}
		}
			
			
		/* ArrayList<SheepDog> livestock = new ArrayList<SheepDog>();
		 System.out.println("Thread creation");
		for(int i=0;i<25;i++){
			SheepDog Dog = new SheepDog(Lines.subList(i*Lines.size()/25, (i+1)*Lines.size()/25),l_filepath);
			livestock.add(Dog);
			Dog.start();
		}
		System.out.println("Thread joining");
		for(SheepDog Dog: livestock){
	    	Dog.join();
	    	}
		*/
		
		File SM = new File(l_filepath +"/SortedMap/"+ "SM_" + number);
		FileUtils.deleteQuietly(SM);
		SM.createNewFile();
		
		ArrayList<String> tmp = new ArrayList<String>();
		for(String key : livestock.keySet()){
			tmp.add(key + " " + StringUtils.join(livestock.get(key),","));
		}
		/*for(SheepDog Dog: livestock){
			String ecriture = StringUtils.join(Dog.result(),"\n");
			FileUtils.write(SM, ecriture);
		}
		*/
		String ecriture = StringUtils.join(tmp,"\n");
		FileUtils.write(SM, ecriture);
		System.out.println("Thread Shuffler " + new Integer(number).toString() + " ends" );
	}
}

