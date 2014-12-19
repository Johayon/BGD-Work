import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;

public class Mapper {
	
	public static void main(String[] args) throws IOException{
		
		/* recuperation des arguments */
		String l_filepath = args[2];
		int MapperID = Integer.parseInt(args[0]);
		int l_N_Slave = Integer.parseInt(args[1]);
		System.out.println("Thread Mapper " + new Integer(MapperID).toString() + " begins" );
		/* Ouverture du split fichier specifique */
		File fichier = new File(l_filepath + "/Split/S_" + new Integer(MapperID).toString());
		List<String> Lines = FileUtils.readLines(fichier, "UTF-8");
		
		/* Creation des Lignes des input partitionnees */
		
		ArrayList<ArrayList<String>> LineInputs = new ArrayList<ArrayList<String>>();
		for(int i=0;i<l_N_Slave;i++){
			LineInputs.add( new ArrayList<String>());
		}
		HashSet<String> key_words = new HashSet<String>();
		
		/* Recuperation des mots dans le split */
		for(String line : Lines){
			String[] Words = line.split("\\s+");
			for(String word : Words){
				LineInputs.get(Math.abs(word.hashCode() % l_N_Slave)).add(word + " 1");
				key_words.add(word);
			}
		}
		
		/* Creation des fichiers UM_k_Ri */
		for(int i=0;i<l_N_Slave;i++){
			File UM = new File(l_filepath +"/UnsortedMap/"+ "UM_" + new Integer(MapperID).toString() + "R_" + new Integer(i).toString());
			FileUtils.deleteQuietly(UM);
			UM.createNewFile();
			String ecriture = StringUtils.join(LineInputs.get(i),"\n");
			FileUtils.write(UM,ecriture);
		}
		
		/*  Creation du fichier cle*/
		File keys = new File(l_filepath +"/Keys/"+ "keys_" + new Integer(MapperID).toString());
		FileUtils.deleteQuietly(keys);
		keys.createNewFile();
		String ecriture2 = StringUtils.join(key_words,"\n");
		FileUtils.write(keys,ecriture2);
		
		System.out.println("Thread Mapper " + new Integer(MapperID).toString() + " ends");
	}
}
