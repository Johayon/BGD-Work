import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;


public class Reducer {
	
	public static void main(String[] args) throws IOException{
		String l_filepath = args[1];
		String number = args[0];
		System.out.println("Thread Reducer " + new Integer(number).toString() + " begins" );
		File fichier = new File(l_filepath + "/SortedMap/SM_" + number);
		List<String> Lines = FileUtils.readLines(fichier, "UTF-8");
		
		File red = new File(l_filepath +"/ReducedMap/"+ "red_" + number);
		FileUtils.deleteQuietly(red);
		red.createNewFile();
		ArrayList<String> LineInput = new ArrayList<String>();
		for(String line : Lines){
			String[] split = line.split(" ");
			String[] count = split[1].split(",");
			Integer result= new Integer(0);
				for(String numb : count){ 
					result += Integer.parseInt(numb);
				}
			LineInput.add(split[0] + " "+ result.toString());
		}
		String ecriture = StringUtils.join(LineInput,"\n");
		FileUtils.write(red,ecriture);	
		System.out.println("Thread Reducer " + new Integer(number).toString() + " begins" );
		
	}

}
